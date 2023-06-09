import torch
from torch import nn, optim
from transformers import GPT2TokenizerFast, GPT2Tokenizer, BertTokenizer
from util_fucntions import util_functions
#from lightning.pytorch.trainer import Trainer
#from lightning.pytorch.callbacks.early_stopping import EarlyStopping

MODEL_ITERATION = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f'Running on {DEVICE}')
INIT_LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 1e-6

LABEL_KEY_NAME = 'harmonised_desc'
INPUT_KEY_NAME = 'curr_desc'
SAMPLE_IDX_CODE_NAME = 'idx'


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
PATIENCE = 4
DROPOUT = 0.1
#EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_mae", patience=PATIENCE, mode="min")
#MY_TRAINER = Trainer(callbacks=[EARLY_STOPPING_CALLBACK])

# Model config
HIDDEN_LAYER_SIZE = 16

NUM_WORKERS = 0
NUM_EPOCHS = 4000
VAL_EPOCH = 20
SAVED_EPOCH = 400
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 4

RANDOM_SEED = 10
TRAIN_RATIO = .65
VALIDATION_RATIO = .2
TEST_RATIO = .15
DATA_FILE_PATH = '../../../xlsx_resources/for_trainings/maximo_pm_to_gap_pm_desc_map.xlsx'

WANDB_PROJECT_NAME = f'Maximo to GAP App harmonised description BERT Base Uncased Model Training Iter{MODEL_ITERATION}'
RUNNING_LOG_LOCATION = f'saved_logs/bert_based_uncased_run_logs/running_iteration_{MODEL_ITERATION}.txt'
SAVED_MODEL_LOCATION = f'saved_models/bert_based_uncased_models_iteration_{MODEL_ITERATION}/'
SAVED_MODEL_FORMAT = '.pt'

SAVED_UNTRAINED_SAMPLE_IDX_LOCATION = f'saved_untrained_wos/bert_based_uncased/running_iteration_{MODEL_ITERATION}.json'
SAVED_TRAINED_SAMPLE_IDX_LOCATION = f'saved_trained_wos/bert_based_uncased/running_iteration_{MODEL_ITERATION}.json'
