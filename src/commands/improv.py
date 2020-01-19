
import music21
from fastai.text import *
from pathlib import Path
import os
from src.utils.midi_encode import Track, fetch_encoder
import torch
from src.models.rnn import RNN
import torch.nn.functional as F

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


# def generate_seq(start, learner, length):
#     seq = start
#     while len(seq.split(' ')) < length:
#         seq = learner.predict(seq)
#     return seq

def predict_topk(prompt, model, vocab, seq_len, k=1):
    seq = prompt.split(',')
    ids = vocab.numericalize(seq)
    vocab_sz = len(vocab.itos)
    model.eval()

    with torch.no_grad():
        while len(ids) < length:
            decoded, hidden = model.forward(torch.LongTensor([ids]), None)
            last_out = F.softmax(decoded.view(-1, vocab_sz)[-1])
            choice = torch.multinomial(torch.tensor(
                torch.topk(last_out, 1)[1], dtype=torch.float), 1)
            highest_prob_idx = int(choice)
            ids.append(highest_prob_idx)
    return vocab.textify(ids)


def predict_nucleus(prompt, model, vocab, seq_len):
    pass


PREDICT_CHOICES = {
    'topk': predict_topk,
    'nucleus': predict_nucleus
}


def generate_seq(prompt, vocab, model, length):
    seq = prompt.split(',')
    ids = vocab.numericalize(seq)
    vocab_sz = len(vocab.itos)
    model.eval()

    with torch.no_grad():
        while len(ids) < length:
            decoded, hidden = model.forward(torch.LongTensor([ids]), None)
            last_out = F.softmax(decoded.view(-1, vocab_sz)[-1])
            topk = torch.tensor(torch.topk(last_out, 1)[
                1], dtype=torch.float)

            choice_idx = int(torch.multinomial(topk, 1))
            ids.append(int(topk[choice_idx]))
    return vocab.textify(ids).split(' ')


def run(args):
    model_dir, fname = os.path.split(args.model)
    encoder = fetch_encoder(args.encoder)()
    print('Loading model...')
    model = RNN(65, 300, 600, 4)
    model.load_state_dict(torch.load(args.model))

    print('Loading vocab...')
    vocab = Vocab.load(Path(model_dir)/'vocab.pkl')

    enc_seq = generate_seq(args.prompt, vocab, model, args.seq)

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
