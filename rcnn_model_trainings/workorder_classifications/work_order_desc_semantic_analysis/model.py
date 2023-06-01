from configs import PRETRAINED_MODEL_NAME, DROPOUT
import torch
from torch import nn, optim
from transformers import BertModel, GPT2Model
from util_fucntions import util_functions

is_bert = 'bert' in util_functions.lower_case_and_clear_white_space(PRETRAINED_MODEL_NAME)


class SentenceSimilarityModel(nn.Module):
    def __init__(self, dropout_rate=DROPOUT):
        super(SentenceSimilarityModel, self).__init__()
        if is_bert:
            self.bert = BertModel.from_pretrained(
                PRETRAINED_MODEL_NAME)
            self.dropout = nn.Dropout(dropout_rate)
            self.hidden_layer = nn.Linear(self.bert.config.hidden_size, 16)
        else:
            self.gpt = GPT2Model.from_pretrained(PRETRAINED_MODEL_NAME)
            self.dropout = nn.Dropout(dropout_rate)
            self.hidden_layer = nn.Linear(self.gpt.config.hidden_size, 16)
        self.relu = nn.ReLU()
        self.regression_layer = nn.Linear(16, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        hidden_output = self.hidden_layer(pooled_output)
        hidden_output = self.relu(hidden_output)
        similarity_score = self.regression_layer(hidden_output)
        return similarity_score
