import numpy as np
import torch

from segmentation_models_pytorch.utils import losses as basic_losses

from segmentation_models_pytorch import losses

from typing import Optional

from torch.nn import functional as F


class ContinuityLoss(torch.nn.Module):
    def __init__(self, initial_loss, stepsize_con: float = 1.0) -> None:
        super().__init__()
        self.stepsize_con = stepsize_con
        self.continuityLoss = torch.nn.L1Loss(size_average=True)
        self.initial_loss = initial_loss
        self.__name__ = initial_loss.__name__ + "_continuity"

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # continuity loss definition
        input = input.type(torch.FloatTensor).cuda()

        lhpy = self.continuityLoss(input[:, :, 1:, :], input[:, :, 0:-1, :])
        lhpz = self.continuityLoss(input[:, :, :, 1:], input[:, :, :, 0:-1])

        return self.initial_loss(input, target)+self.stepsize_con*(lhpy + lhpz)


class CategoricalCrossEntropyLoss(basic_losses.CrossEntropyLoss):

    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', mode=None) -> None:
        super(CategoricalCrossEntropyLoss, self).__init__(
            weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        # self.__name__="categorical_crossentropy"

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _target = torch.argmax(target, dim=1)
        return F.cross_entropy(input, _target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


def my_losses():
    # aliases imported from segmentation_models.losses:
    my_dict = {"jaccard_loss": losses.jaccard.JaccardLoss,
               "dice_loss": losses.dice.DiceLoss,
               "categorical_crossentropy": CategoricalCrossEntropyLoss,
               "focal_loss": losses.focal.FocalLoss,
               "continuity_loss": ContinuityLoss
               }
    return my_dict
