import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from configs import *
from model import SentenceSimilarityModel
from dataset import get_splitted_dataset, get_data_loaders


def model_param_tweaking(model) -> tuple:
    loss_func = nn.MSELoss(reduction=MSE_REDUCTION)
    optimiser = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=.1, patience=PATIENCE,
                                                                 threshold=.0001, threshold_mode='abs')

    return loss_func, optimiser, train_scheduler


def train(train_dataloader, model, loss_func, optimiser, epoch: int):
    size = len(train_dataloader.dataset)
    total_loss = 0
    model.train()
    for batch_inputs, batch_targets in train_dataloader:
        # Forward pass
        outputs = model(batch_inputs)

        # Compute the loss
        loss = loss_func(outputs, batch_targets)

        # Backward pass and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Accumulate the epoch loss
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {avg_loss:.4f}")


def validate(val_dataloader, model, loss_func, epoch: int):
    model.eval()
    total_loss = 0
    with torch.no_grad() if torch.cuda.is_available() else True:
        for val_batch_inputs, val_batch_targets in val_dataloader:
            # Forward pass
            val_outputs = model(val_batch_inputs)
            # Compute the validation loss
            total_loss += loss_func(val_outputs, val_batch_targets).item()

        avg_loss = total_loss / len(val_dataloader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation Loss: {avg_loss:.4f}")


def main():
    model = SentenceSimilarityModel().to(DEVICE)
    loss_func, optimiser, train_scheduler = model_param_tweaking(model)
    train_set, val_test, test_set = get_splitted_dataset()
    train_loader, validation_loader, test_loader = get_data_loaders(train_set, val_test, test_set)
    for epoch in range(NUM_EPOCHS):
        write_info = "___Epoch {}______________________________________________________________________".format(epoch + 1)
        print(write_info)
        if (epoch + 1) % VAL_EPOCH == 0:
            validate(val_dataloader=validation_loader, model=model, epoch=epoch, loss_func=loss_func)
        else:
            train(train_dataloader=train_loader, model=model, epoch=epoch, loss_func=loss_func, optimiser=optimiser)


if __name__ == '__main__':
    main()
