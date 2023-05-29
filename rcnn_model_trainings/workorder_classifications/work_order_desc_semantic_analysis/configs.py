import torch
from torch import nn, optim
from transformers import BertModel, BertTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f'Running on {DEVICE}')
INIT_LEARNING_RATE = 2e-5

PRETRAINED_MODEL_NAME = 'bert-base-uncased'
BERT_TOKENIZER = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

MSE_REDUCTION = 'mean'
WEIGHT_DECAY = .0001
SCHEDULED = True
PATIENCE = 16


NUM_EPOCHS = 10000
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 8

RANDOM_SEED = 10
TRAIN_RATIO = .65
VALIDATION_RATIO = .2
TEST_RATIO = .15
DATA_FILE_PATH = '../../../xlsx_resources/for_trainings/rct_pm_desc_similarity.xlsx'
