import torch
from torch import nn, optim
from transformers import GPT2TokenizerFast, GPT2Tokenizer, BertTokenizer
from util_fucntions import util_functions

MODEL_ITERATION = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f'Running on {DEVICE}')
INIT_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 2e-5

ACTUAL_VALUE_KEY_NAME = 'similarity'
TEXT1_KEY_NAME = 'rct_desc'
TEXT2_KEY_NAME = 'pm_wo_desc'
SAMPLE_IDX_CODE_NAME = 'mapping_code'


PRETRAINED_MODEL_NAME = 'bert-base-uncased'

IS_BERT = 'bert' in util_functions.lower_case_and_clear_white_space(PRETRAINED_MODEL_NAME)
MODEL_TOKENIZER = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME) if IS_BERT else GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
PADDING_TOKEN = None if IS_BERT else '[PAD]'
if PADDING_TOKEN is not None:
    MODEL_TOKENIZER.add_special_tokens({'pad_token': PADDING_TOKEN})
    padding_token_id = MODEL_TOKENIZER.encode(PADDING_TOKEN)[0]

MAX_LENGTH_TOKEN = 128

MSE_REDUCTION = 'mean'
WEIGHT_DECAY = .0001
SCHEDULED = True
PATIENCE = 16
DROPOUT = 0.1

# Model config
HIDDEN_LAYER_SIZE = 16

NUM_WORKERS = 0
NUM_EPOCHS = 2000
VAL_EPOCH = 10
SAVED_EPOCH = 200
TRAIN_BATCH_SIZE = 44
VAL_BATCH_SIZE = 4

RANDOM_SEED = 10
TRAIN_RATIO = .65
VALIDATION_RATIO = .2
TEST_RATIO = .15
DATA_FILE_PATH = '../../../xlsx_resources/for_trainings/Book5.xlsx'

WANDB_PROJECT_NAME = f'BERT Base Uncased Model Training Iter{MODEL_ITERATION}'
RUNNING_LOG_LOCATION = f'saved_logs/bert_based_uncased_run_logs/running_iteration_{MODEL_ITERATION}.txt'
SAVED_MODEL_LOCATION = f'saved_models/bert_based_uncased_models_iteration_{MODEL_ITERATION}/'
SAVED_MODEL_FORMAT = '.pt'

SAVED_UNTRAINED_SAMPLE_IDX_LOCATION = f'saved_untrained_wos/bert_based_uncased/running_iteration_{MODEL_ITERATION}.json'
SAVED_TRAINED_SAMPLE_IDX_LOCATION = f'saved_trained_wos/bert_based_uncased/running_iteration_{MODEL_ITERATION}.json'
