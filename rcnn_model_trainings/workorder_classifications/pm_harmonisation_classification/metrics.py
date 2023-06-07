import torch
import torch.nn as nn
from torch import Tensor
import torchmetrics
import numpy as np
from torchmetrics import F1Score


def cal_rmse(y_true, y_pred) -> Tensor:
    eps = 1e-6
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(y_true, y_pred) * eps)


def cal_mae(y_true, y_pred) -> Tensor:
    # Calculate Mean Absolute Error
    mae = nn.L1Loss()
    return mae(y_true, y_pred).item()


def _to_torch_tensor(data):
    return torch.tensor(data)


def cal_accuracy(y_true, y_pred) -> float:
    return torchmetrics.functional.accuracy(_to_torch_tensor(y_pred), _to_torch_tensor(y_true)).item()


def cal_precision(y_true, y_pred) -> float:
    return torchmetrics.functional.precision(_to_torch_tensor(y_pred), _to_torch_tensor(y_true)).item()


def cal_recall(y_true, y_pred) -> float:
    return torchmetrics.functional.recall(_to_torch_tensor(y_pred), _to_torch_tensor(y_true)).item()


def cal_f1_scores(y_true, y_pred, num_classes) -> dict:
    return {
        'micro': torchmetrics.functional.f1_score(_to_torch_tensor(y_pred), _to_torch_tensor(y_true), num_classes=num_classes, average='mirco').item(),
        'macro': torchmetrics.functional.f1_score(_to_torch_tensor(y_pred), _to_torch_tensor(y_true),
                                                  num_classes=num_classes, average='macro').item(),
    }
