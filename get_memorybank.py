import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn as nn
import clip
from clip.InfoNCELoss import InfoNCELoss
from torch import optim
from util import set_seed_logger, get_logger
from params import parse_args
from scheduler import cosine_lr
from eval import evaluate, evaluate_1K
from data import prepare_dataloader
import copy
from torch import autograd
from torch.cuda.amp import autocast as autocast
from sklearn.mixture import GaussianMixture
import numpy as np
global logger

def get_mbank(noise_ratio, dataset_name, sim, text_index, image_feature, text_feature, model, dataloader):
    '''
    model: pre-trained clip or your own model
    noise_ratio: the proportion of noise in the dataset, for example, 0, 0.2, and 0.4.
    dataset_name: choose in {MSCOCO, Flickr30K, CC120K}
    '''
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        clean_img = []
        clean_cap = []

        text_index = torch.cat(text_index)
        image_feature = torch.cat(image_feature)
        text_feature = torch.cat(text_feature)

        arg_c = np.argwhere((sim>30) == True)

        clean_idx = text_index[arg_c].cpu()

        image_feature = image_feature.cpu().numpy()
        text_feature = text_feature.cpu().numpy()

        clean_img = image_feature[clean_idx]
        clean_cap = text_feature[clean_idx]

    mbank_img_idx = {}
    mbank_txt_idx = {}

    clean_img = torch.tensor(clean_img).squeeze().cuda()
    clean_cap = torch.tensor(clean_cap).squeeze().cuda()
    clean_idx = np.array(clean_idx)
    clean_idx = clean_idx.squeeze()

    print(clean_img.shape)
    print(clean_cap.shape)
    print(clean_idx.shape)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader) :
            if idx%20 == 0:
                logger.info("Calculating loss for all samples: %d/%d", idx, len(dataloader))
            images, img_ids, texts, txt_ids = batch #get data
            # texts, txt_ids = batch
            
            images = images.cuda()
            texts = texts.cuda()
            #------ clip ------
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            image_sim = image_features @ clean_img.t()
            txt_sim = text_features @ clean_cap.t()
            #---------------------------------------------------------------------------------------------
            img_max_sim, img_max_idx = torch.topk(image_sim, k=1, dim=1, largest = True)
            txt_max_sim, txt_max_idx = torch.topk(txt_sim, k=1, dim=1, largest = True)
            #---------------------------------------------------------------------------------------------
            img_max_idx = img_max_idx.cpu()
            txt_max_idx = txt_max_idx.cpu()
            m_img_idx = clean_idx[img_max_idx]
            m_txt_idx = clean_idx[txt_max_idx]
            #------------------------------------------------------------------------------------------
            for i in range(len(img_ids)):
                mbank_img_idx[str(int(img_ids[i]))] = m_img_idx[i]

            for i in range(len(txt_ids)):
                mbank_txt_idx[str(int(txt_ids[i]))] = m_txt_idx[i]
        #----------------------------------------------------------------------------------------------------------------------------
        np.save('dataset/{}/annotations/query_bank/{}_mbank_img_idx.npy'.format(dataset_name, str(noise_ratio)), mbank_img_idx)
        np.save('dataset/{}/annotations/query_bank/{}_mbank_txt_idx.npy'.format(dataset_name, str(noise_ratio)), mbank_txt_idx)




def eval_train_gmm(args, model, train_dataloader, train_length, epoch):
    model.eval()
    logits = torch.zeros(train_length)

    text_index = []
    image_feature = []
    text_feature = []
    
    for idx, batch in enumerate(train_dataloader):
        images, texts, _ = batch
        images = images.cuda()
        texts = texts.cuda()
        text_index.append(_)
        with torch.no_grad():
            logits_per_image, logits_per_text, image_features, text_features = model(images, texts)
            image_feature.append(image_features)
            text_feature.append(text_features)
            logits_mean = (torch.diag(logits_per_image) + torch.diag(logits_per_text))/2
            for i, ids in enumerate(_):
                logits[ids] = logits_mean[i]

        if (idx % args.display == 0) and (idx != 0):
            logger.info("eval_train step:%d/%d", idx, len(train_dataloader))
    
    sim = logits.clone()
    input_logits = logits.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_logits.cpu().numpy())
    prob = gmm.predict_proba(input_logits.cpu().numpy())
    prob = prob[:, gmm.means_.argmax()]
    prob = torch.tensor(prob)

    prob = (prob - prob.min()) / (prob.max() - prob.min())

    return sim, prob, text_index, image_feature, text_feature











def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def set_requires_grad(net: nn.Module, mode=True):
    for p in net.parameters():
        p.requires_grad_(mode)

def main():
    global logger
    args = parse_args()

    seed = set_seed_logger(args)
    dir_path = os.path.join(args.checkpoint_path, args.experiments)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    logger = get_logger(os.path.join(dir_path, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    model_clip, preprocess = clip.clip.load(args.vision_model, device=device, jit=False) #Must set jit=False for training

    if args.resume:
        checkpoint = torch.load(args.resume)
        model = model_clip
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model from {}".format(args.resume))

    else:
        model = model_clip 
        logger.info("Model Initialized!")

    model = model.cuda()

    if args.eval:
        train_dataloader = None
        train_length = 0
        args.epochs = 0
        dataloader = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'test')
        test_dataloader, test_length = dataloader['test']

        if args.dataset == 'coco':
            eval_Rank = evaluate(args, model, test_dataloader, logger)
            eval_Rank_1K = evaluate_1K(args, model, test_dataloader, logger) # Only for MSCOCO 1K testing
        else:
            eval_Rank = evaluate(args, model, test_dataloader, logger)
    
    else:
        dataloader = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'eval_train')
        train_dataloader, train_length = dataloader['eval_train']
        dataloader_dev = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'dev')
        dev_dataloader, dev_length = dataloader_dev['dev']

    loss = InfoNCELoss()
    loss = loss.cuda()

    total_steps = train_length * args.epochs

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", total_steps)

    sim, prob, text_index, image_feature, text_feature = eval_train_gmm(args, model, train_dataloader, train_length, 1)

    mb_bank_dataloader = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'mb_bank')
    mb_bank_dataloader, mb_bank_train_length = mb_bank_dataloader['mb_bank']
    get_mbank(args.noise_ratio, args.dataset, sim, text_index, image_feature, text_feature, model, mb_bank_dataloader)

if __name__ == '__main__':
    main()