from configs import PRETRAINED_MODEL_NAME, DROPOUT, HIDDEN_LAYER_SIZE
import torch
from torch import nn, optim
from transformers import BertModel, GPT2Model
from util_fucntions import util_functions

is_bert = 'bert' in util_functions.lower_case_and_clear_white_space(PRETRAINED_MODEL_NAME)


class SentenceSimilarityModel(nn.Module):
    def __init__(self, dropout_rate=DROPOUT):
        super(SentenceSimilarityModel, self).__init__()
        self.model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME) if is_bert else GPT2Model.from_pretrained(PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, HIDDEN_LAYER_SIZE),
            nn.LeakyReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        similarity_score = self.fc(pooled_output)

        return similarity_score
