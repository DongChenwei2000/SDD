import math
import copy
import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):

    def __init__(self, logit_scale=100):
        super(InfoNCELoss, self).__init__()
        self.logit_scale = logit_scale

    def forward(self, input, target, reduction=None): 
        exp = torch.exp(input)
        tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()
        exp_2 = torch.exp(input)
        tmp2 = exp_2.sum(1)
        softmax = tmp1 / tmp2
        log = -torch.log(softmax)

        if reduction == "mean": 
            return log.mean()
        else:
            return log