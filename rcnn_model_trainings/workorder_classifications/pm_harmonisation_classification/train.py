import torch
import wandb
from configs import *
from model import TextClassification
from dataset import get_splitted_dataset, get_data_loaders
from util_fucntions import util_functions
from metrics import *
import copy
from transformers import AdamW
from tqdm import tqdm
import torch.nn.functional as F


def get_learning_rate(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']


def wandb_running_log(loss, accuracy, precision, recall, f1_macro, f1_micro, state="Train"):
    wandb.log({f'{state}/Loss': loss, f'{state}/Accuracy': accuracy, f'{state}/Precision': precision,
               f'{state}/Recall': recall, f'{state}/F1 Macro': f1_macro, f'{state}/F1 Micro': f1_micro})


def write_training_config(num_trains: int, num_vals: int, num_tests: int, classes: list, model: TextClassification):
    all_classes_to_str = ', '.join(classes)
    _min_lr_stmt = f'Min learning rate {MIN_LEARNING_RATE}\n' if MIN_LEARNING_RATE > 0 else ''
    _saved_log = f"\t{util_functions.get_formatted_today_str(twelve_h=True)}\n" \
                 f"Dataset directory: {DATA_FILE_PATH}\nScheduled Learning: {SCHEDULED}\nLearning rate: {INIT_LEARNING_RATE}\n{_min_lr_stmt}" \
                 f"Dropout: {DROPOUT}\nWeight decay: {WEIGHT_DECAY}\nPatience: {PATIENCE}\n" \
                 f"Number of running epochs: {NUM_EPOCHS}\nValidate after every {VAL_EPOCH}th epoch\n" \
                 f"Train-Validation-Test ratio: {TRAIN_RATIO}-{VALIDATION_RATIO}-{TEST_RATIO}\n" \
                 f"Number of train - validation - test samples: {num_trains} - {num_vals} - {num_tests}\n" \
                 f"Train batch size: {TRAIN_BATCH_SIZE}\nValidation batch size: {VAL_BATCH_SIZE}\n" \
                 f"Max length token: {MAX_LENGTH_TOKEN}\nModel name: {PRETRAINED_MODEL_NAME}\n" \
                 f"Model's Full Connect Layer Structure:{model.fc}\n" \
                 f"Running log location: {RUNNING_LOG_LOCATION}\nModel location: {SAVED_MODEL_LOCATION}\n" \
                 f"{len(classes)} Class{'es' if len(classes) > 1 else ''}: {all_classes_to_str}" \
                 f"\n_______________________________________________________________________________________\n"
    util_functions.save_running_logs(_saved_log, RUNNING_LOG_LOCATION)


def model_param_tweaking(model) -> tuple:
    loss_func = nn.CrossEntropyLoss()
    optimiser = AdamW(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if MIN_LEARNING_RATE > 0:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=.1, patience=PATIENCE,
                                                                  min_lr=MIN_LEARNING_RATE, verbose=True)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=.1, patience=PATIENCE,
                                                                  verbose=True)
    return loss_func, optimiser, lr_scheduler


def _get_forward_pass(batch, model):
    return model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))


def run_model(dataloader, model, loss_func, optimiser, is_train=True) -> tuple:
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    total_f1_macro = 0
    total_f1_micro = 0
    model.train() if is_train else model.eval()

    for batch in tqdm(dataloader):
        true_labels = dataloader.dataset.get_label_index(batch[LABEL_KEY_NAME])
        # Forward pass
        predicted_label_probs = _get_forward_pass(batch, model)

        '''if len(predicted_labels.shape) != len(true_labels.shape):
            true_labels = true_labels.squeeze()
            predicted_labels = predicted_labels.squeeze()'''

        # Compute the loss
        # print(f'actuals: {true_labels}')
        # print(f'predicted: {predicted_label_probs}')
        loss = F.cross_entropy(predicted_label_probs, true_labels)
        if DEVICE == 'mps':
            loss = loss.type(torch.float32)

        if is_train and optimiser is not None:
            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # Accumulate the epoch loss
        total_loss += loss.item()
        _, predicted_labels = torch.max(predicted_label_probs, 1)
        # print(f'predicteds: {predicted_labels}')
        total_accuracy += cal_accuracy(true_labels, predicted_labels)
        total_recall += cal_recall(true_labels, predicted_labels)
        total_precision += cal_precision(true_labels, predicted_labels)
        f1_scores = cal_f1_scores(true_labels, predicted_labels, model.get_num_classes())
        total_f1_macro += f1_scores['macro']
        total_f1_macro += f1_scores['micro']

    return tuple([score / len(dataloader) for score in
                 [total_loss, total_accuracy, total_recall, total_precision, total_f1_macro, total_f1_micro]])


def train(train_dataloader, model, loss_func, optimiser, epoch: int) -> tuple:
    avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1_macro, avg_f1_micro = run_model(
        dataloader=train_dataloader, model=model, loss_func=loss_func,
        optimiser=optimiser)
    print('\n')
    util_functions.save_running_logs(
        f'Epoch [{epoch + 1}/{NUM_EPOCHS}] Average train Loss: {avg_loss:.4f}\tAverage Accuracy: {round(avg_accuracy * 100, 2)}%\tAverage Precision: {avg_precision:.4f}\t'
        f'Average Recall: {avg_recall:.4f}\tAverage F1 Macro: {avg_f1_macro:.4f}\tAverage F1 Micro: {avg_f1_micro:.4f}\t'
        f'Learning rate: {get_learning_rate(optimiser)}',
        RUNNING_LOG_LOCATION)
    wandb_running_log(loss=avg_loss, accuracy=avg_accuracy, precision=avg_precision, recall=avg_recall,
                      f1_macro=avg_f1_macro, f1_micro=avg_f1_micro)
    return avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1_macro, avg_f1_micro


def validate(val_dataloader, model, loss_func, optimiser, epoch: int) -> tuple:
    def _process_model_validation() -> tuple:
        avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1_macro, avg_f1_micro = run_model(
            dataloader=val_dataloader, model=model, loss_func=loss_func,
            optimiser=optimiser, is_train=False)
        print('\n')
        util_functions.save_running_logs(
            f'Validation Epoch [{epoch + 1}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f}\tAverage Accuracy: {round(avg_accuracy * 100, 2)}%\tAverage Precision: {avg_precision:.4f}\t'
            f'Average Recall: {avg_recall:.4f}\tAverage F1 Macro: {avg_f1_macro:.4f}\tAverage F1 Micro: {avg_f1_micro:.4f}\t',
            RUNNING_LOG_LOCATION)
        wandb_running_log(loss=avg_loss, accuracy=avg_accuracy, precision=avg_precision, recall=avg_recall,
                          f1_macro=avg_f1_macro, f1_micro=avg_f1_micro, state='Validation')
        return avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1_macro, avg_f1_micro

    if torch.cuda.is_available():
        with torch.no_grad():
            return _process_model_validation()
    else:
        return _process_model_validation()


def test(test_dataloader, model, loss_func):
    if torch.cuda.is_available():
        with torch.no_grad():
            avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1_macro, avg_f1_micro = run_model(
                dataloader=test_dataloader, model=model, is_train=False,
                loss_func=loss_func, optimiser=None)
    else:
        avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1_macro, avg_f1_micro = run_model(
            dataloader=test_dataloader, model=model, is_train=False,
            loss_func=loss_func, optimiser=None)
    print('\n')
    util_functions.save_running_logs(
        f'Test --- Average Loss: {avg_loss:.4f}\tAverage Accuracy: {round(avg_accuracy * 100, 2)}%\tAverage Precision: {avg_precision:.4f}\t'
        f'Average Recall: {avg_recall:.4f}\tAverage F1 Macro: {avg_f1_macro:.4f}\tAverage F1 Micro: {avg_f1_micro:.4f}\t',
        RUNNING_LOG_LOCATION)
    wandb_running_log(loss=avg_loss, accuracy=avg_accuracy, precision=avg_precision, recall=avg_recall,
                      f1_macro=avg_f1_macro, f1_micro=avg_f1_micro, state='Test')


def main():
    train_set, val_set, test_set, _classes = get_splitted_dataset()
    train_loader, validation_loader, test_loader = get_data_loaders(train_set, val_set, test_set, _classes)

    static_classes = train_loader.dataset.classes
    model = TextClassification(num_classes=len(static_classes)).to(DEVICE)
    best_model = copy.deepcopy(model)
    # MY_TRAINER.fit(model)

    loss_func, optimiser, lr_scheduler = model_param_tweaking(model)
    best_accuracy = 0

    write_training_config(len(train_set), len(val_set), len(test_set), _classes, model)

    for epoch in range(NUM_EPOCHS):
        wandb.log({'Train/lr': get_learning_rate(optimiser)})

        if epoch > 1 and (epoch + 1) % VAL_EPOCH == 0:
            avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1_macro, avg_f1_micro = validate(
                val_dataloader=validation_loader, model=model, epoch=epoch,
                loss_func=loss_func,
                optimiser=optimiser)
            # model.validation_step(avg_mae)
            if avg_accuracy < best_accuracy:
                best_accuracy = avg_accuracy
                best_model = copy.deepcopy(model)
                util_functions.save_running_logs(f'\tCurrent best model at epoch {epoch + 1}', RUNNING_LOG_LOCATION)
                util_functions.save_model(model=best_model,
                                          optimiser=optimiser, loss=avg_loss, epoch=epoch,
                                          saved_location=f"{SAVED_MODEL_LOCATION}best_model{SAVED_MODEL_FORMAT}")
        else:
            avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1_macro, avg_f1_micro = train(
                train_dataloader=train_loader, model=model, epoch=epoch,
                loss_func=loss_func, optimiser=optimiser)
            if epoch > 1 and (epoch + 1) % SAVED_EPOCH == 0:
                util_functions.save_model(model=model, optimiser=optimiser, loss=avg_loss, epoch=epoch,
                                          saved_location=f'{SAVED_MODEL_LOCATION}model_epoch{epoch}{SAVED_MODEL_FORMAT}')
                util_functions.save_running_logs(
                    f'-----------Save model at epoch [{epoch + 1}/{NUM_EPOCHS}] at {SAVED_MODEL_LOCATION} -----------',
                    RUNNING_LOG_LOCATION)

        if SCHEDULED and epoch >= VAL_EPOCH:
            lr_scheduler.step(best_accuracy)

    util_functions.save_running_logs('Training complete, running final testing:', RUNNING_LOG_LOCATION)
    test(test_dataloader=test_loader, model=model, loss_func=loss_func)
    util_functions.save_model(model=model, optimiser=optimiser,
                              saved_location=f'{SAVED_MODEL_LOCATION}final_model{SAVED_MODEL_FORMAT}')

    util_functions.save_running_logs('Testing with best VAL model:', RUNNING_LOG_LOCATION)
    test(test_dataloader=test_loader, model=best_model, loss_func=loss_func)


if __name__ == '__main__':
    wandb.init(project=WANDB_PROJECT_NAME)
    main()
    wandb.finish()
