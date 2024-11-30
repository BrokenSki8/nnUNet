import torch
from torch import nn, Tensor
from typing import List

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class CVLoss(nn.Module):
    def __init__(self, idx_start:List[int]):
        super(CVLoss, self).__init__()
        # first start of the second dataset, then the indices of the first image of each dataset
        self.idx_start = idx_start

    def forward(self, idx: int, input: Tensor, target: Tensor):
        # Example loss computation

        dataset_idx = self.dataset_idx(idx)
        if dataset_idx == 0:
            return self.compute_loss_0(input, target)
        elif dataset_idx == 1:
            return self.compute_loss_1(input, target)
        return -1
    
    def dataset_idx(self, idx: int):
        #TODO : implement from which dataset the image is from
        for i in range(len(self.idx_start)):
            if idx < self.idx_start[i]:
                return i
        return 0 # default to 0
    
    def compute_loss_0(self, input: Tensor, target: Tensor):
        # Map every input that is not 0 to 1
        input = torch.where(input != 0, torch.ones_like(input), input)

        # compute loss 
        loss = self.dice_loss(input, target)

        return loss
    
    def compute_loss_1(self, input: Tensor, target: Tensor):
        # no need to mao to anything just yet

        # compute loss 
        loss = self.dice_loss(input, target)

        return loss
    
    def dice_loss(self, input: Tensor, target: Tensor):
        shp_x = input.shape
        axes = list(range(2, len(shp_x)))
        tp, fp, fn, _ = get_tp_fp_fn_tn(input, target, axes, False)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + 1e-5) / (torch.clip(denominator + 1e-5, 1e-8))

        dc = dc.mean()
        
        return -dc