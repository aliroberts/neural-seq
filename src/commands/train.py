
from src.utils.system import ensure_dir_exists


import errno
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


def construct_language_model(arch: Callable, vocab_sz: int, config: dict = None, drop_mult: float = 1.):
    "Create a language model from `arch` and its `config`, maybe `pretrained`."
    meta = {'hid_name': 'emb_sz', 'url': URLs.WT103_FWD, 'url_bwd': URLs.WT103_BWD,
            'config_lm': awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
            'config_clas': awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split}
    config = ifnone(config, meta['config_lm']).copy()
    for k in config.keys():
        if k.endswith('_p'):
            config[k] *= drop_mult
    tie_weights, output_p, out_bias = map(
        config.pop, ['tie_weights', 'output_p', 'out_bias'])
    init = config.pop('init') if 'init' in config else None

    encoder = arch(vocab_sz, **config)
    enc = encoder.encoder if tie_weights else None
    decoder = LinearDecoder(
        vocab_sz, config[meta['hid_name']], output_p, tie_encoder=enc, bias=out_bias)
    model = SequentialRNN(encoder, decoder)
    return model if init is None else model.apply(init)


class MIDISeqTokenizer(BaseTokenizer):
    def tokenizer(self, text):
        return text.split(',')


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

    # TODO: Come up with extensible way of adding new models
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
    data_lm = TextLMDataBunch.from_folder(
        args.data_dir, tokenizer=CustomTokenizer(), include_bos=False,
        include_eos=False, bs=args.bs, vocab=vocab, bptt=int(args.bptt))
    # import ipdb
    # ipdb.set_trace()
    print('Saving vocab...')
    data_lm.vocab.save(dest_dir/'vocab.pkl')
    print('Training...')

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

    # TODO: Save model state for resuming training (figure out how to organise this between model file and this one)

    # learn = LanguageLearner(
    #     data_lm, model, model_dir=dest_dir)

    # model = construct_language_model(
    #     RNN, len(data_lm.vocab.itos), config=awd_lstm_lm_config.update(config))
    # learn = LanguageLearner(
    #     data_lm, model, model_dir=dest_dir)

    learn = language_model_learner(
        data_lm, AWD_LSTM, model_dir=dest_dir, drop_mult=args.drop_mult, pretrained=False, config=awd_lstm_lm_config.update(config))
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
    else:
        latest_epoch = 0

    epochs = args.epochs
    save_freq = args.save_freq if args.save_freq > 0 else epochs

    for i in range(epochs):
        learn.fit_one_cycle(1, args.max_lr, moms=(
            0.8, 0.7), start_epoch=latest_epoch)
        if (i + 1) % save_freq == 0:
            save_path = f'{args.prefix}-epoch-{i + 1 + latest_epoch}'
            export_path = f'{args.prefix}-epoch-{i + 1 + latest_epoch}-export'
            print('Saving model checkpoint to', save_path)
            learn.save(save_path)
            print('Exporting model checkpoint to', export_path)
            learn.export(export_path)
    print('DONE')
