import torch
from torch import nn, Tensor
from typing import List
import torch.nn.functional as F

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class CVLoss(nn.Module):
    def __init__(self, idx_start:List[int]):
        super(CVLoss, self).__init__()
        # first start of the second dataset, then the indices of the first image of each dataset
        self.idx_start = idx_start

    def forward(self, input: Tensor, target: Tensor):
        # Example loss computation

        dataset_idx = self.dataset_idx(target)
        if dataset_idx == 0:
            return self.compute_loss_0(input, target)
        elif dataset_idx == 1:
            return self.compute_loss_1(input, target)
        
        raise ValueError("Dataset index not recognized")
    
    def dataset_idx(self, target: Tensor):

        # MAPPING DICTIONARIES TO LABELS
        # ct_to_final = {'1': 1, # myocardium
        #     '2': 2, # left atrium
        #     '3': 3, # left ventricle
        #     '4': 4, # right atrium
        #     '5': 5, # right ventricle
        #     '6': 6, # aorta
        #     '7': 7, # pulmonary veins
        #     '10': 8, # pulmonary artery
        #     '15': 9, # superior vena cava
        #     '16': 10, # inferior vena cava
        #     '17': 11} # portal vein and splenic vein
        # mri_to_final = {'1': 1, # myocardium
        #     '2': 3, # left ventricle
        #     '3': 5, # right ventricle
        #     '4': 2, # left atrium
        #     '5': 4, # right atrium
        #     '6': 8, # pulmonary artery
        #     '7': 6, # ascending aorta --> mapped to aorta
        #     '8': 6} # descending aorta --> mapped to aorta
        # mri_1label_to_final = {'1': 12} # whole heart label
        # TODO : implement from which dataset the image is from
       
        # Count number of unique labels in the target
        nb_unique_labels = len(torch.unique(target))

        if nb_unique_labels == 12: # might be 12 if we include the background
            return 2
        
        if nb_unique_labels == 9: # might be 7 if we include the background
            return 1

        if nb_unique_labels == 2:
            return 0
        
        if nb_unique_labels >= 3: # need to do cahnge that to make sure  we choose the right one
            return 1
        raise ValueError("Number of unique labels not recognized")
        # return 0 # default to 0
    
    def compute_loss_0(self, input: Tensor, target: Tensor):
        # Map every input that is not 1, 2, 3, 4, 5, 12 to 1 
        map_to_dataset_0 = {0: 0, 1: 1, 2: 1, 3: 1,
                            4: 1, 5: 1, 6: 0, 7: 0,
                            8: 0, 9: 0, 10: 0, 11: 0,
                            12: 1}
        
        input_copy = input
        for k, v in map_to_dataset_0.items():
            input[input_copy == k] = v
        
        # compute loss 
        loss = self.dice_loss(input, target)

        return loss
 
    
    def compute_loss_1(self, input: Tensor, target: Tensor):
        # no need to map to anything just yet

        # compute loss 
        loss = self.dice_loss(input, target)

        return loss
    
    def dice_loss(self, input: Tensor, target: Tensor):
        shp_x = input.shape
        axes = list(range(2, len(shp_x)))
        tp, fp, fn, _ = get_tp_fp_fn_tn(input, target, axes, None, False)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + 1e-5) / (torch.clip(denominator + 1e-5, 1e-8))

        dc = dc.mean()
        
        return -dc
    
    def chat_maxrule(self, logits, candidate_label_mask):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes), raw scores from the model.
            candidate_label_mask: Tensor of shape (batch_size, num_classes), 
                                  binary mask indicating candidate labels (1 for candidate, 0 otherwise).
        Returns:
            loss: Scalar tensor, the partial label loss.
        """

        # TODO: check what the imputes are, I think they already done the softmax but have to chekc the code
         
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)
        
        # Mask out non-candidate labels
        masked_probabilities = probabilities * candidate_label_mask  # Shape: (batch_size, num_classes)
        
        # Sum probabilities over candidate labels
        candidate_sums = masked_probabilities.sum(dim=1)  # Shape: (batch_size,)
        
        # Avoid log(0) by clamping probabilities
        candidate_sums = candidate_sums.clamp(min=1e-9)
        
        # Compute the negative log likelihood loss
        loss = -torch.log(candidate_sums).mean()  # Average over the batch
        
        return loss