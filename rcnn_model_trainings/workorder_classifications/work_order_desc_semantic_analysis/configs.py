import torch
from torch import nn, optim
from transformers import GPT2TokenizerFast, BertTokenizer
from util_fucntions import util_functions

MODEL_ITERATION = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f'Running on {DEVICE}')
INIT_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 2e-5

ACTUAL_VALUE_KEY_NAME = 'similarity'
TEXT1_KEY_NAME = 'rct_desc'
TEXT2_KEY_NAME = 'pm_wo_desc'
SAMPLE_IDX_CODE_NAME = 'mapping_code'


PRETRAINED_MODEL_NAME = 'gpt2'

MODEL_TOKENIZER = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME) if 'bert' in util_functions.lower_case_and_clear_white_space(PRETRAINED_MODEL_NAME) else GPT2TokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
PADDING_TOKEN = '[PAD]'
if type(PADDING_TOKEN) is str and not PADDING_TOKEN.isspace():
    MODEL_TOKENIZER.add_tokens([PADDING_TOKEN])
    padding_token_id = MODEL_TOKENIZER.encode(MODEL_TOKENIZER)[0]

MAX_LENGTH_TOKEN = 256

MSE_REDUCTION = 'mean'
WEIGHT_DECAY = .0001
SCHEDULED = True
PATIENCE = 16
DROPOUT = 0.1

NUM_WORKERS = 0
NUM_EPOCHS = 1500
VAL_EPOCH = 10
SAVED_EPOCH = 300
TRAIN_BATCH_SIZE = 18
VAL_BATCH_SIZE = 4

RANDOM_SEED = 10
TRAIN_RATIO = .65
VALIDATION_RATIO = .2
TEST_RATIO = .15
DATA_FILE_PATH = '../../../xlsx_resources/for_trainings/Book5.xlsx'

WANDB_PROJECT_NAME = f'GPT2 Model Training Iter{MODEL_ITERATION}'
RUNNING_LOG_LOCATION = f'saved_logs/gpt2_run_logs/running_iteration_{MODEL_ITERATION}.txt'
SAVED_MODEL_LOCATION = f'saved_models/gpt2_models_iteration_{MODEL_ITERATION}/'
SAVED_MODEL_FORMAT = '.pt'

SAVED_UNTRAINED_SAMPLE_IDX_LOCATION = f'saved_untrained_wos/gpt2/running_iteration_{MODEL_ITERATION}.json'
SAVED_TRAINED_SAMPLE_IDX_LOCATION = f'saved_trained_wos/gpt2/running_iteration_{MODEL_ITERATION}.json'
