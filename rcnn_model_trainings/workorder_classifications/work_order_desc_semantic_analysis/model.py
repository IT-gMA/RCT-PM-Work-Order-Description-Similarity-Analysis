from configs import PRETRAINED_MODEL_NAME
import torch
from torch import nn, optim
from transformers import BertModel, BertTokenizer


class SentenceSimilarityModel(nn.Module):
    def __init__(self):
        super(SentenceSimilarityModel, self).__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.regression_layer = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        similarity_score = self.regression_layer(pooled_output)
        return similarity_score
