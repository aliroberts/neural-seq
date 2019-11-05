
import errno
import os
import random
import shutil

from fastai.text import * 

DATA_DIR_ALL = Path('data/encoded')
OUT_DIR = Path('data/')

def ensure_dir_exists(path):
    """ Ensure that a directory at the given path exists """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class MIDISeqTokenizer(BaseTokenizer):
    def tokenizer(self, text):
        return text.split(',')
    
class CustomTokenizer(Tokenizer):
    "Put together rules and a tokenizer function to tokenize text with multiprocessing."
    def __init__(self, lang:str='en', pre_rules:ListRules=None,
                 post_rules:ListRules=None, special_cases:Collection[str]=None, n_cpus:int=None):
        super().__init__(tok_func=MIDISeqTokenizer, pre_rules=[], post_rules=[], special_cases=[])

def main():
    RANDOM_SEED = 42
    DATA_DIR = OUT_DIR
    print('Loading data...')
    data_lm = load_data(DATA_DIR, 'data_lm_export.pkl')
    print('Training...')
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5, pretrained=False)
    learn.fit_one_cycle(10, 5e-3, moms=(0.8,0.7))
    learn.save('model_v1_10_epochs')

if __name__ == '__main__':
    main()