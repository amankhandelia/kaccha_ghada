from transformers import AutoTokenizer, RobertaTokenizerFast, RobertaTokenizer
import pandas as pd
from utils import save_readable_vocab, get_readable_tokenization, get_identity_tokenization

models_df = pd.read_csv('/home/khandelia1000/kaccha_ghada/hindi/mlm_test_config.csv')

models_df['from_flax'] = models_df['from_flax']
models_df['use_fast'] = models_df['use_fast']
models_df['add_prefix_space'] = models_df['add_prefix_space']
example_texts = ['क', 'संसद', 'ममता', 'मानवता', 'गौशाला', 'मधुशाला', 'मेरी मम्मी तो मुझे पुरे दिन भर मारती ही रहती ऐसी मम्मी कहीं नहीं मिलती']
outputs = []
for i in range(len(models_df)):
    from_flax = models_df.iloc[i]['from_flax'].item()
    use_fast = models_df.iloc[i]['use_fast'].item()
    add_prefix_space = models_df.iloc[i]['add_prefix_space'].item()
    model_name = models_df.iloc[i]['model_name']
    readable_csv_path = '/home/khandelia1000/tokens/{}_vocab.csv'.format(model_name.split('/')[1])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, add_prefix_space=add_prefix_space, from_flax=from_flax)
    output = []
    for text in example_texts:
        output.append((text, get_readable_tokenization(text, tokenizer), get_identity_tokenization(text, tokenizer)))
    outputs.append((model_name, output))
pd.DataFrame(outputs).to_csv('/home/khandelia1000/tokens/hindi_tokens.csv')

