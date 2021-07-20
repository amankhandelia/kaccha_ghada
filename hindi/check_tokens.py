from transformers import AutoTokenizer, RobertaTokenizerFast, RobertaTokenizer
import pandas as pd
from utils import save_readable_vocab, tokenize_readable

model_name, readable_csv_path = 'surajp/RoBERTa-hindi-guj-san', '/home/khandelia1000/tokens/suraj_hindi_vocab.csv'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
example_texts = ['संसद', 'ममता', 'मानवता', 'गौशाला', 'मधुशाला']
for text in example_texts:
    print(tokenize_readable(text, tokenizer))

