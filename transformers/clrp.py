import transformers

from transformers import pipeline
from transformers import AutoTokenizer, AutoMode

import pandas as pd

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)