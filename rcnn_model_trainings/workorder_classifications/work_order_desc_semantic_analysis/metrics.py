import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


def cal_rmse(y_true: float, y_pred: float) -> Tensor:
    eps = 1e-6
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(y_true, y_pred) * eps)


def cal_mae(y_true: float, y_pred: float) -> Tensor:
    # Calculate Mean Absolute Error
    mae = nn.L1Loss()
    return mae(y_true, y_pred).item()
