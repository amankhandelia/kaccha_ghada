from transformers import AutoTokenizer, RobertaTokenizer, RobertaTokenizerFast
import pandas as pd
from typing import List


def save_readable_vocab(tokenizer, readable_csv_path):
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

def get_readable_merges(merges:dict, tokenizer:AutoTokenizer):
    readable_merges = {}
    for tokens, idx in merges.items():
        readable_tokens = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
        tokens = tuple(readable_tokens)
        readable_merges[tokens] = idx
    return readable_merges

def get_identity_tokenization(text:str, tokenizer:AutoTokenizer):
    tokens = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_string(tokens)

def test_tokenization(model_name:str, from_flax:bool, use_fast:bool, add_prefix_space:bool, example_texts:List):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, add_prefix_space=add_prefix_space, from_flax=from_flax)
    output = []
    for text in example_texts:
        output.append((text, get_readable_tokenization(text, tokenizer), get_identity_tokenization(text, tokenizer)))
    return output

def devnagri_unicode_block():
    devnagri_block_len = 128
    devnagri_block_start = 2304
    devnagri_chars = [chr(i) for i in range(devnagri_block_start, devnagri_block_start + devnagri_block_len)]
    return devnagri_chars

def get_all_devnagri_char_by_category(category=None):
    matra = ['ँ', 'ं', 'ः', '़', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॅ', 'े', 'ै', 'ॉ', 'ो', 'ौ', '्']
    rare_matra = ['ऀ', 'ॖ', 'ॗ', 'ॢ', 'ॣ', 'ॎ', 'ॏ', '॑', '॒', '॓', '॔', 'ॕ', 'ऺ', 'ऻ', 'ॊ', 'ॆ']
    swara = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ऍ', 'ऎ', 'ए', 'ऐ', 'ऑ', 'ओ', 'औ']
    rare_vowels = ['ऄ', 'ऒ', 'ऌ']
    
    vyanjan = ['क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'ळ', 'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'ॠ']
    consonants_with_nukta = ['ऩ', 'ऱ', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 'य़']
    sanyukta_akshar = ['क्ष','त्र','ज्ञ','श्र']
    rare_characters = ['ॡ', 'ॾ', 'ॿ', 'ॲ', 'ॳ', 'ॴ', 'ॵ', 'ॶ', 'ॷ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ']
    digits = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
    symbols = ['।', '॥', 'ॽ', 'ॐ', 'ऽ', '॰', 'ॱ']
    if category == 'matras':
        return matra
    elif category == 'vowels':
        return swara
    elif category == 'consonants':
        return vyanjan
    elif category == 'digits':
        return digits
    elif category == 'rare_characters':
        return rare_characters
    elif category == 'symbols':
        return symbols
    elif category == 'consonants_with_nukta':
        return consonants_with_nukta
    elif category == 'sanyukta_akshar':
        return sanyukta_akshar
    elif category == 'rare_vowels':
        return rare_vowels
    elif category == 'rare_matra':
        return rare_matra
    else:
        return (matra, swara, vyanjan, consonants_with_nukta, rare_characters, digits, symbols)