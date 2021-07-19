import transformers

from transformers import pipeline
from transformers import AutoTokenizer

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

class CLRP(nn.Module):
    def __init__(self, model_name, hidden_size):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(hidden_size, 1)
        

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

if __name__ == '__main__':
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    model = CLRP(model_name, 768).to(xm.xla_device())