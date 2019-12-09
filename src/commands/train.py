
from src.utils.system import ensure_dir_exists


import errno
import os
from pathlib import Path
import random
import shutil

from src.utils.system import ensure_dir_exists

from fastai.text import *

DATA_DIR_ALL = Path('data/encoded')
OUT_DIR = Path('data/')


class MIDISeqTokenizer(BaseTokenizer):
    def tokenizer(self, text):
        return text.split(',')


class CustomTokenizer(Tokenizer):
    "Put together rules and a tokenizer function to tokenize text with multiprocessing."

    def __init__(self, lang: str = 'en', pre_rules: ListRules = None,
                 post_rules: ListRules = None, special_cases: Collection[str] = None, n_cpus: int = None):
        super().__init__(tok_func=MIDISeqTokenizer,
                         pre_rules=[], post_rules=[], special_cases=[])


def run(args):
    # TODO: Come up with extensible way of adding new models
    dest_dir = Path(os.path.abspath(args.dest))
    ensure_dir_exists(dest_dir)
    print('Loading data...')
    data_lm_path = dest_dir/'data_lm_export.pkl'

    if args.cache:
        data_lm = TextLMDataBunch.from_folder(
            args.data_dir, tokenizer=CustomTokenizer(), include_bos=False,
            include_eos=False, bs=args.bs)
        data_lm.save(data_lm_path)
    else:
        try:
            data_lm = load_data(args.data_dir, data_lm_path)
        except FileNotFoundError:
            print('No pickled databunch found, generating a new one')
            data_lm = TextLMDataBunch.from_folder(
                args.data_dir, tokenizer=CustomTokenizer(), include_bos=False,
                include_eos=False, bs=args.bs)
            data_lm.save(data_lm_path)

    print('Training...')
    learn = language_model_learner(
        data_lm, AWD_LSTM, model_dir=dest_dir, drop_mult=args.drop_mult, pretrained=False, config=awd_lstm_lm_config.update({
            'emb_sz': args.emb_sz,
            'n_hid': args.nhid,
            'n_layers': args.nlayers
        }))
    learn.path = dest_dir

    if args.pretrained:
        learn.load(args.pretrained)

    if args.resume:

        def extract_epochs(dir):
            epochs = []
            for fname in os.listdir(dir):
                match = re.search(args.prefix + '-epoch-(\d+)', fname)
                if match:
                    epochs.append(int(match.group(1)))
            epochs.sort()
            return epochs

        epochs_found = extract_epochs(dest_dir)
        if epochs_found:
            latest_epoch = epochs_found[-1]
            fname = f'{args.prefix}-epoch-{latest_epoch}'
            print(f'Latest model found matching prefix: ', fname)
            learn.load(fname)
        else:
            latest_epoch = 0
            print('No pretrained models found')

    epochs = args.epochs
    save_freq = args.save_freq if args.save_freq > 0 else epochs

    for i in range(epochs):
        learn.fit_one_cycle(1, args.max_lr, moms=(0.8, 0.7))
        if (i + 1) % save_freq == 0:
            save_path = f'{args.prefix}-epoch-{i + 1 + latest_epoch}'
            export_path = f'{args.prefix}-epoch-{i + 1 + latest_epoch}-export'
            print('Saving model checkpoint to', save_path)
            learn.save(save_path)
            print('Exporting model checkpoint to', export_path)
            learn.export(export_path)
    print('DONE')
