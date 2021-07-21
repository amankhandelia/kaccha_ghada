from transformers import AutoTokenizer
from utils import get_readable_tokenization, save_readable_vocab

def test_tokenize_readable(tokenizer, test_cases = None):
    if not test_cases:
        test_cases = [('संसद', ['स', 'ं', 'सद'])]
    for test_case, gt in test_cases:
        test_tokens = get_readable_tokenization(test_case, tokenizer)
        assert test_tokens == gt, 'tokenize_readable({}) = {}, expected {}'.format(test_case, test_tokens, gt)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('flax-community/roberta-pretraining-hindi', use_fast=True)
    test_tokenize_readable(tokenizer)
    