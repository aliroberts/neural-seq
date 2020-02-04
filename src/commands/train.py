
from src.utils.system import ensure_dir_exists


import errno
import json
import os
from pathlib import Path
import pickle
import random
import shutil

from src.utils.models import fetch_model
from src.utils.system import ensure_dir_exists

from fastai.text import *

DATA_DIR_ALL = Path('data/encoded')
OUT_DIR = Path('data/')


class MIDISeqTokenizer(BaseTokenizer):
    def tokenizer(self, text):
        return json.loads(text)['data']


class CustomTokenizer(Tokenizer):
    "Put together rules and a tokenizer function to tokenize text with multiprocessing."

    def __init__(self, lang: str = 'en', pre_rules: ListRules = None,
                 post_rules: ListRules = None, special_cases: Collection[str] = None, n_cpus: int = None):
        super().__init__(tok_func=MIDISeqTokenizer,
                         pre_rules=[], post_rules=[], special_cases=[])


def run(args, model_kwargs, train_kwargs):
    """
    Accept command line args as a param and the model and train kwargs (dicts with {<kwarg_name>: <default_val>,...})
    """
    dest_dir = Path(os.path.abspath(args.dest))
    ensure_dir_exists(dest_dir)
    print('Loading data...')
    data_lm_path = dest_dir/'data_lm_export.pkl'

    with open(Path(args.data_dir)/'vocab', 'r') as f:
        vocab = Vocab(f.read().split(','))

    # if args.cache:
    #     data_lm = TextLMDataBunch.from_folder(
    #         args.data_dir, tokenizer=CustomTokenizer(), include_bos=False,
    #         vocab=vocab,
    #         include_eos=False, bs=args.bs, bptt=args.bptt)
    #     data_lm.save(data_lm_path)
    # else:
    #     try:
    #         data_lm = load_data(args.data_dir, data_lm_path)
    #     except FileNotFoundError:
    #         print('No pickled databunch found, generating a new one')
    #         data_lm = TextLMDataBunch.from_folder(
    #             args.data_dir, tokenizer=CustomTokenizer(), include_bos=False,
    #             include_eos=False, bs=args.bs, vocab=vocab, bptt=args.bptt)
    #         data_lm.save(data_lm_path)

    if 'test' in os.listdir(args.data_dir):
        test = 'test'
    else:
        test = None

    data_lm = TextLMDataBunch.from_folder(
        args.data_dir, tokenizer=CustomTokenizer(), include_bos=False,
        include_eos=False, bs=args.bs, vocab=vocab, bptt=int(args.bptt), test=test)
    # import ipdb
    # ipdb.set_trace()
    print('Saving vocab...')
    data_lm.vocab.save(dest_dir/'vocab.pkl')

    model_custom_kwargs = {k: getattr(args, k)
                           for k, _ in model_kwargs.items()}
    train_custom_kwargs = {k: getattr(args, k)
                           for k, _ in train_kwargs.items()}

    Model, train_func, _ = fetch_model(args.model)

    model = Model(len(data_lm.vocab.itos), **model_custom_kwargs)

    with open(dest_dir/'params.pkl', 'wb') as f:
        pickle.dump(model_custom_kwargs, f)

    train_func(data_lm, model, args, **train_custom_kwargs)
    return
