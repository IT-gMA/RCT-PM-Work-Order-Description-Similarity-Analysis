from configs import PRETRAINED_MODEL_NAME, DROPOUT, HIDDEN_LAYER_SIZE, IS_BERT
import torch
from torch import nn, optim
from transformers import BertModel, GPT2Model
from util_fucntions import util_functions
import torch.nn.functional as F


class TextClassification(nn.Module):
    def __init__(self, num_classes: int, dropout_rate=DROPOUT):
        super(TextClassification, self).__init__()
        self.num_classes = num_classes
        self.model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME) if IS_BERT else GPT2Model.from_pretrained(PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        logits = self.fc(pooled_output)
        return F.softmax(logits, dim=1)

    def get_num_classes(self) -> int:
        return self.num_classes

    def set_num_classes(self, num_classes: int) -> None:
        self.num_classes = num_classes
