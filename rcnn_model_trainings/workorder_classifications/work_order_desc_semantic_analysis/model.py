from configs import PRETRAINED_MODEL_NAME, DROPOUT, HIDDEN_LAYER_SIZE, IS_BERT
import torch
from torch import nn, optim
from transformers import BertModel, GPT2Model
from util_fucntions import util_functions


class SentenceSimilarityModel(nn.Module):
    def __init__(self, dropout_rate=DROPOUT):
        super(SentenceSimilarityModel, self).__init__()
        self.model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME) if IS_BERT else GPT2Model.from_pretrained(PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) if IS_BERT else self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        similarity_score = self.fc(pooled_output)

        return similarity_score

    def validation_step(self, val_mae):
        self.log('val_mae', val_mae)
