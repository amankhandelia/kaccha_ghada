from transformers import AutoTokenizer, RobertaTokenizer, RobertaTokenizerFast
import pandas as pd


def save_readable_vocab(model_name, readable_csv_path, use_fast=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    if isinstance(tokenizer, RobertaTokenizer):
        tokens_df = pd.DataFrame({'tokens':list(tokenizer.encoder.keys())})
    else:
        tokens_df = pd.DataFrame({'tokens':list(tokenizer.vocab.keys())})
    tokens_df['tokens'] = tokens_df['tokens'].map(lambda x: tokenizer.convert_tokens_to_string([x]))
    tokens_df.to_csv(readable_csv_path, index=False, header=False)
    return

def get_readable_tokenization(text:str, tokenizer:AutoTokenizer):
    tokens = tokenizer.tokenize(text)
    readable_tokens = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    return readable_tokens

def get_identity_tokenization(text:str, tokenizer:AutoTokenizer):
    tokens = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_string(tokens)


   

    