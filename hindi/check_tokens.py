from transformers import AutoTokenizer, RobertaTokenizerFast, RobertaTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import pandas as pd
from utils import get_readable_merges, get_readable_tokenization, get_all_devnagri_char_by_category


if __name__ == '__main__':
    # model_name, readable_csv_path = 'flax-community/roberta-base-mr', '/home/khandelia1000/tokens/suraj_hindi_vocab.csv'
    model_name = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # example_texts = ['संसद', 'ममता', 'मानवता', 'गौशाला', 'मधुशाला']
    example_texts = ['flexibility is keeey to moronic behaviourism']
    for text in example_texts:
        print(get_readable_tokenization(text, tokenizer))

    
    
    
        
    


        