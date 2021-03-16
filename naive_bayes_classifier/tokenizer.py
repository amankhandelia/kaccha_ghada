import re
from typing import List

def dumb_tokenize(text:str) -> List[str]:
    # a very dumb tokenizer that splits the text on white space
    tokens = text.split(' ')
    return tokens
