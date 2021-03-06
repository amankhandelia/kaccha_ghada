import pandas as pd
import re, string
from utils import test_tokenization, save_readable_vocab, get_garbage_data
from datasets.iterable_dataset import IterableDataset


def run_test_tokenization(config_file = '/home/khandelia1000/kaccha_ghada/hindi/mlm_test_config.csv', output_file='/home/khandelia1000/tokens/hindi_tokens.csv'):
    
    example_texts = ['क', 'संसद', 'ममता', 'मानवता', 'गौशाला', 'मधुशाला', 'मेरी मम्मी तो मुझे पुरे दिन भर मारती ही रहती ऐसी मम्मी कहीं नहीं मिलती']
    outputs = []
    
    # read models csv and remove irrelavent columns
    models_df = pd.read_csv(config_file)
    del models_df['display_name'], models_df['revision']
    
    for i in range(len(models_df)):
        kwargs = models_df.iloc[i].to_dict()
        output = test_tokenization(**kwargs, example_texts=example_texts)
        outputs.append((kwargs['model_name'], output))
    
    # save outputs to csv
    pd.DataFrame(outputs).to_csv(output_file)

def save_all_vocabs():
    # read models csv and remove irrelavent columns
    models_df = pd.read_csv(config_file)
    del models_df['display_name'], models_df['revision']
    
    for i in range(len(models_df)):
        kwargs = models_df.iloc[i].to_dict()
        tokenizer = AutoTokenizer.from_pretrained(**kwargs)
        readable_csv_path = '/home/khandelia1000/tokens/{}_vocab.csv'.format(kwargs['model_name'].split('/')[1])
        save_readable_vocab(tokenizer, readable_csv_path)

def get_all_garbage_text(dataset:IterableDataset, max_docs:int = 1000):
    processed_docs_count = 0
    contains_garbage_count = 0
    for example in dataset:
        processed_docs_count += 1
        garbage_text, is_just_punctuation = get_garbage_data(example['text'])
        if not is_just_punctuation:
            if len(garbage_text) > 0:
                contains_garbage_count += 1
                # print only english text
                english_with_punct = re.compile("[" + " a-zA-Z0-9" +re.escape(string.punctuation) + string.digits + "]")
                garbage_text = re.sub(r"[ ]+", " ", "".join([tok.group() for tok in re.finditer(english_with_punct, garbage_text)]))
                # print(garbage_text)
        if processed_docs_count > max_docs:
            break
    print("Processed {} docs, found garbage in {} docs".format(processed_docs_count, contains_garbage_count))

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("mc4", "hi", split="train", streaming=True)
    get_all_garbage_text(dataset)
    
    
    
    
        
    


        