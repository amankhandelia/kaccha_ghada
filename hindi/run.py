import pandas as pd
from utils import test_tokenization, save_readable_vocab

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

if __name__ == '__main__':
    run_test_tokenization()
    
    
    
    
        
    


        