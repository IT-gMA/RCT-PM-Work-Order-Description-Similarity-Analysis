import torch
import wandb
from configs import *
from model import SentenceSimilarityModel
from dataset import get_splitted_dataset, get_data_loaders
from util_fucntions import util_functions
from metrics import cal_rmse, cal_mae
import copy


def get_learning_rate(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']


def wandb_running_log(loss, mae, rmse, state="Train"):
    wandb.log({f'{state}/loss': loss, f'{state}/rmse': rmse, f'{state}/mape': mae})


def write_training_config(num_trains: int, num_vals: int, num_tests: int):
    _min_lr_stmt = f'Min learning rate {MIN_LEARNING_RATE}\n' if MIN_LEARNING_RATE > 0 else ''
    _saved_log = f"\t{util_functions.get_formatted_today_str(twelve_h=True)}\n" \
                 f"Dataset directory: {DATA_FILE_PATH}\nScheduled Learning: {SCHEDULED}\nLearning rate: {INIT_LEARNING_RATE}\n{_min_lr_stmt}" \
                 f"Dropout: {DROPOUT}\nWeight decay: {WEIGHT_DECAY}\nPatience: {PATIENCE}\n" \
                 f"Number of running epochs: {NUM_EPOCHS}\nValidate after every {SAVED_EPOCH}th epoch\n" \
                 f"MSE Reduction: {MSE_REDUCTION}\nTrain-Validation-Test ratio: {TRAIN_RATIO}-{VALIDATION_RATIO}-{TEST_RATIO}\n" \
                 f"Number of train - validation - test samples: {num_trains} - {num_vals} - {num_tests}\n" \
                 f"Train batch size: {TRAIN_BATCH_SIZE}\nValidation batch size: {VAL_BATCH_SIZE}\n" \
                 f"Max length token: {MAX_LENGTH_TOKEN}\nModel name: {PRETRAINED_MODEL_NAME}\n" \
                 f"Running log location: {RUNNING_LOG_LOCATION}\nModel location: {SAVED_MODEL_LOCATION}" \
                 f"_________________________________________________________\n"
    util_functions.save_running_logs(_saved_log, RUNNING_LOG_LOCATION)


def model_param_tweaking(model) -> tuple:
    loss_func = nn.MSELoss(reduction=MSE_REDUCTION)
    optimiser = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if MIN_LEARNING_RATE > 0:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=.1, patience=PATIENCE,
                                                                  min_lr=MIN_LEARNING_RATE)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=.1, patience=PATIENCE)
    return loss_func, optimiser, lr_scheduler


def run_model(dataloader, model, loss_func, optimiser, is_train=True) -> tuple:
    total_loss = 0
    total_rmse = 0
    total_mae = 0
    model.train() if is_train else model.eval()

    for batch in dataloader:
        target_similarity_scores = batch['similarity'].to(DEVICE)  # actual similarity scores
        if target_similarity_scores.shape > 1:
            target_similarity_scores = target_similarity_scores.squeeze()

        outputs = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['token_type_ids'].to(DEVICE))  # Forward pass
        if outputs.shape > 1:
            outputs = outputs.squeeze()

        # Compute the loss
        loss = loss_func(outputs, target_similarity_scores)
        if DEVICE == 'mps':
            loss = loss.type(torch.float32)

        if is_train and optimiser is not None:
            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # Accumulate the epoch loss
        total_loss += loss.item()
        total_mae += cal_mae(target_similarity_scores, outputs)
        total_rmse += cal_rmse(target_similarity_scores, outputs)

    return total_loss / len(dataloader), total_mae / len(dataloader), total_rmse / len(dataloader)


def train(train_dataloader, model, loss_func, optimiser, epoch: int) -> tuple:
    avg_loss, avg_mae, avg_rmse = run_model(dataloader=train_dataloader, model=model, loss_func=loss_func,
                                            optimiser=optimiser)
    util_functions.save_running_logs(
        f'Epoch [{epoch + 1}/{NUM_EPOCHS}] Average train Loss: {avg_loss:.4f}\tAverage RMSE: {avg_rmse:.4f}\tAverage MAE: {avg_mae:.4f}\tLearning rate: {get_learning_rate(optimiser):.4f}',
        RUNNING_LOG_LOCATION)
    wandb_running_log(loss=avg_loss, mae=avg_mae, rmse=avg_rmse)
    return avg_loss, avg_mae, avg_rmse


def validate(val_dataloader, model, loss_func, optimiser, epoch: int) -> tuple:
    def _process_model_validation() -> tuple:
        avg_loss, avg_mae, avg_rmse = run_model(dataloader=val_dataloader, model=model, loss_func=loss_func,
                                                optimiser=optimiser, is_train=False)
        util_functions.save_running_logs(
            f'Epoch [{epoch + 1}/{NUM_EPOCHS}] Average validation Loss: {avg_loss:.4f}\tAverage RMSE: {avg_rmse:.4f}\tAverage MAE: {avg_mae:.4f}\tLearning rate: {get_learning_rate(optimiser):.4f}',
            RUNNING_LOG_LOCATION)
        wandb_running_log(loss=avg_loss, mae=avg_mae, rmse=avg_rmse, state='Validation')
        return avg_loss, avg_mae, avg_rmse

    if torch.cuda.is_available():
        with torch.no_grad():
            return _process_model_validation()
    else:
        return _process_model_validation()


def test(test_dataloader, model, loss_func):
    if torch.cuda.is_available():
        with torch.no_grad():
            avg_loss, avg_mae, avg_rmse = run_model(dataloader=test_dataloader, model=model, is_train=False,
                                                    loss_func=loss_func, optimiser=None)
    else:
        avg_loss, avg_mae, avg_rmse = run_model(dataloader=test_dataloader, model=model, is_train=False,
                                                loss_func=loss_func, optimiser=None)
    util_functions.save_running_logs(
        f'Average test Loss: {avg_loss:.4f}\tAverage RMSE: {avg_rmse:.4f}\tAverage MAE: {avg_mae:.4f}',
        RUNNING_LOG_LOCATION)
    wandb_running_log(loss=avg_loss, mae=avg_mae, rmse=avg_rmse, state='Test')


def main():
    model = SentenceSimilarityModel().to(DEVICE)
    best_model = copy.deepcopy(model)

    loss_func, optimiser, lr_scheduler = model_param_tweaking(model)
    train_set, val_set, test_set = get_splitted_dataset()
    train_loader, validation_loader, test_loader = get_data_loaders(train_set, val_set, test_set)
    best_mae = 10

    write_training_config(len(train_set), len(val_set), len(test_set))
    for epoch in range(NUM_EPOCHS):
        wandb.log({'Train/lr': get_learning_rate(optimiser)})

        if epoch > 1 and (epoch + 1) % VAL_EPOCH == 0:
            avg_loss, avg_mae, avg_rmse = validate(val_dataloader=validation_loader, model=model, epoch=epoch,
                                                   loss_func=loss_func,
                                                   optimiser=optimiser)
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_model = copy.deepcopy(model)
                util_functions.save_running_logs(f'\tCurrent best model at epoch {epoch + 1}', RUNNING_LOG_LOCATION)
                util_functions.save_model(model=best_model,
                                          optimiser=optimiser, loss=avg_loss, epoch=epoch,
                                          saved_location=f"{SAVED_MODEL_LOCATION}best_model_epoch_{epoch}{SAVED_MODEL_FORMAT}")
        else:
            avg_loss, avg_mae, avg_rmse = train(train_dataloader=train_loader, model=model, epoch=epoch,
                                                loss_func=loss_func, optimiser=optimiser)
            if epoch > 1 and (epoch + 1) % SAVED_EPOCH == 0:
                util_functions.save_model(model=model, optimiser=optimiser, loss=avg_loss, epoch=epoch,
                                          saved_location=f'{SAVED_MODEL_LOCATION}model_epoch{epoch}{SAVED_MODEL_FORMAT}')
                util_functions.save_running_logs(
                    f'-----------Save model at epoch [{epoch + 1}/{NUM_EPOCHS}] at {SAVED_MODEL_LOCATION} -----------',
                    RUNNING_LOG_LOCATION)

        if SCHEDULED and epoch >= VAL_EPOCH:
            lr_scheduler.step(best_mae)

    util_functions.save_running_logs('Training complete, running final testing:', RUNNING_LOG_LOCATION)
    test(test_dataloader=test_loader, model=model, loss_func=loss_func)
    util_functions.save_model(model=model, optimiser=optimiser,
                              saved_location=f'{SAVED_MODEL_LOCATION}final_model{SAVED_MODEL_FORMAT}')

    util_functions.save_running_logs('Testing with best VAL model:', RUNNING_LOG_LOCATION)


if __name__ == '__main__':
    wandb.init(project=WANDB_PROJECT_NAME)
    main()
    wandb.finish()
