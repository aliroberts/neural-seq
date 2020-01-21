
import music21
from fastai.text import *
from pathlib import Path
import os
from src.utils.midi_encode import Track, fetch_encoder
import torch
from src.models.awd_lstm import RNNModel
import torch.nn.functional as F
import random

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


def predict_beam(prompt, model, vocab, args):
    raise NotImplementedError


def predict_topk(prompt, model, vocab, args):
    seq = prompt

    ids = vocab.numericalize(seq)
    vocab_sz = len(vocab.itos)
    model.eval()

    with torch.no_grad():
        while len(ids) < args.seq:
            decoded = model.predict(torch.LongTensor([[ids[-1]]]))
            # decoded, hidden = model.forward(torch.LongTensor([ids]), None)
            last_out = F.softmax(decoded.view(-1, vocab_sz)[-1], dim=0)
            topk = torch.as_tensor(torch.topk(last_out, args.k)[
                1], dtype=torch.float)
            choice_idx = int(torch.multinomial(topk, 1))
            ids.append(int(topk[choice_idx]))
    return vocab.textify(ids).split(' ')


def predict_nucleus(prompt, model, vocab, args):
    raise NotImplementedError


PREDICT_CHOICES = {
    'beam': predict_beam,
    'topk': predict_topk,
    'nucleus': predict_nucleus
}


def run(args):
    model_dir, fname = os.path.split(args.model)
    encoder = fetch_encoder(args.encoder)()
    print('Loading model...')
    model = RNNModel(65, 300, 600, 1)
    model.load_state_dict(torch.load(args.model))

    print('Loading vocab...')
    vocab = Vocab.load(Path(model_dir)/'vocab.pkl')

    predict_func = PREDICT_CHOICES[args.sample]

    # If no prompt has been provided, choose a random token from our vocab to start with
    prompt = args.prompt.split(',') if args.prompt else [
        random.choice(vocab.itos)]

    enc_seq = predict_func(prompt, model, vocab, args)

    # enc_seq = generate_seq(args.prompt, learn, args.seq)
    tokens = encoder.process_prediction(enc_seq) * args.loop

    dec_seq = encoder.decode(tokens)
    stream = music21.stream.Stream()
    t = music21.tempo.MetronomeMark(number=args.tempo)
    stream.append(t)
    for note in dec_seq:
        stream.append(note)

    sp = music21.midi.realtime.StreamPlayer(stream)
    sp.play()
