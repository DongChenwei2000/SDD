# SDD
This is a PyTorch implementation for SDD.

## Dataset

We conducted experiments on three datasets: MSCOCO, Flickr30K, and CC120K. We followed [NPC](https://github.com/ZhangXu0963/NPC) to split image-text pairs in MSCOCO and FLickr30K into training, validation and testing sets. All datasets can be obtained in  [NPC](https://github.com/ZhangXu0963/NPC).

## Training

**For training SDD** You can train a new model via the following command. Before training, you can read the `params.py` carefully to check your parameter setting. The `--num_anns` should be set to 5 for MSCOCO and Flickr30K, and 1 for CC120K.

```python
python main_SDD_sim.py --batch_size 256 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio ${NOISE RATIO} --num_anns ${5 or 1} --dataset_root ${YOUR PATH} --dataset coco --checkpoint_path ${YOUR PATH}
```

## Evaluation

You can use the following code to evaluate your model:

```python
python main_SDD_sim.py --eval --resume ${YOUR PATH} --dataset_root ${YOUR PATH} --dataset coco
```

