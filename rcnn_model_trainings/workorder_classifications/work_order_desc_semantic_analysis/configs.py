import torch
from torch import nn, optim
from transformers import BertModel, BertTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f'Running on {DEVICE}')
INIT_LEARNING_RATE = 2e-3
MIN_LEARNING_RATE = 2e-5

ACTUAL_VALUE_KEY_NAME = 'similarity'
TEXT1_KEY_NAME = 'rct_desc'
TEXT2_KEY_NAME = 'pm_wo_desc'
SAMPLE_IDX_CODE_NAME = 'mapping_code'


PRETRAINED_MODEL_NAME = 'bert-base-uncased'
BERT_TOKENIZER = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
MAX_LENGTH_TOKEN = 256

MSE_REDUCTION = 'mean'
WEIGHT_DECAY = .0001
SCHEDULED = True
PATIENCE = 16
DROPOUT = 0.1

NUM_WORKERS = 0
NUM_EPOCHS = 100
VAL_EPOCH = 5
SAVED_EPOCH = 10
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 4

RANDOM_SEED = 10
TRAIN_RATIO = .65
VALIDATION_RATIO = .2
TEST_RATIO = .15
DATA_FILE_PATH = '../../../xlsx_resources/for_trainings/Book5.xlsx'

WANDB_PROJECT_NAME = 'BERT Based Uncased Iter 01'
RUNNING_LOG_LOCATION = 'saved_logs/bert_based_uncased_run_logs/running_iteration_1.txt'
SAVED_MODEL_LOCATION = 'saved_models/bert_based_uncased_models_iteration_1/'
SAVED_MODEL_FORMAT = '.pt'

SAVED_UNTRAINED_SAMPLE_IDX_LOCATION = 'saved_untrained_wos/bert_based_uncased/running_iteration_1.json'
SAVED_TRAINED_SAMPLE_IDX_LOCATION = 'saved_trained_wos/bert_based_uncased/running_iteration_1.json'
