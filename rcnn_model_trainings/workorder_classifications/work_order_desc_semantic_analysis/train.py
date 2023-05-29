import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from configs import *
from model import SentenceSimilarityModel


def model_param_tweaking(model) -> tuple:
    loss_func = nn.MSELoss(reduction=MSE_REDUCTION)
    optimiser = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=.1, patience=PATIENCE,
                                                                 threshold=.0001, threshold_mode='abs')

    return loss_func, optimiser, train_scheduler


def main():
    model = SentenceSimilarityModel().to(DEVICE)
    loss_func, optimiser, train_scheduler = model_param_tweaking(model)


if __name__ == '__main__':
    main()
