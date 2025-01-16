import torch
import torch.nn as nn


def mse_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor
    ):
    targets = targets.unsqueeze(dim=1)

    loss = nn.MSELoss()(inputs, targets)
    return loss

def mse_loss_new(
    inputs: torch.Tensor,
    targets: torch.Tensor
    ):
    loss = nn.MSELoss()(inputs, targets)
    return loss

def l1_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor
    ):

    loss =  nn.L1Loss()(inputs, targets.unsqueeze(dim=1))
    return loss