
import music21
from fastai.text import *
from pathlib import Path
import pickle
import os
from src.utils.midi_data import play_midi
from src.utils.midi_encode import MIDIData, fetch_encoder
from src.utils.models import fetch_model
import torch
import torch.nn.functional as F
import random

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


def predict_beam(prompt, model, encoder, vocab, args):
    raise NotImplementedError


def predict_topk(prompt, model, encoder, vocab, args):
    seq = prompt

    ids = vocab.numericalize(seq)
    vocab_sz = len(vocab.itos)
    model.eval()
    with torch.no_grad():
        while encoder.duration(vocab.textify(ids).split(' ')) < args.seq:
            decoded = model.predict(torch.LongTensor([[ids[-1]]]))
            # decoded, hidden = model.forward(torch.LongTensor([ids]), None)
            last_out = F.softmax(decoded.view(-1, vocab_sz)[-1], dim=0)
            topk = torch.as_tensor(torch.topk(last_out, args.k)[
                1], dtype=torch.float)
            choice_idx = int(torch.multinomial(topk, 1))
            ids.append(int(topk[choice_idx]))
    return encoder.process_prediction(vocab.textify(ids).split(' '), seq_length=args.seq)


def predict_nucleus(prompt, model, encoder, vocab, args):
    raise NotImplementedError


PREDICT_CHOICES = {
    'beam': predict_beam,
    'topk': predict_topk,
    'nucleus': predict_nucleus
}


def run(args):
    model_dir, fname = os.path.split(args.state)
    encoder = fetch_encoder(args.encoder)()

    print('Loading vocab...')
    vocab = Vocab.load(Path(model_dir)/'vocab.pkl')

    print('Loading model...')
    with open(Path(model_dir)/'params.pkl', 'rb') as f:
        model_kwargs = pickle.load(f)

    Model, _, _ = fetch_model(args.model)
    model = Model(len(vocab.itos), **model_kwargs)

    model.load_state_dict(torch.load(args.state))

    predict_func = PREDICT_CHOICES[args.sample]

    # If no prompt has been provided, choose a random token from our vocab to start with
    prompt = args.prompt.split(',') if args.prompt else [
        random.choice(vocab.itos)]

    enc_seq = predict_func(prompt, model, encoder, vocab, args)

    # enc_seq = generate_seq(args.prompt, learn, args.seq)
    tokens = enc_seq * args.loop

    decoded = encoder.decode(tokens, tempo=args.tempo)
    play_midi(decoded)
