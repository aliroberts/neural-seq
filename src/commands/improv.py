
import music21
from fastai.text import *
from pathlib import Path
import os
from src.utils.midi_encode import Track


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


def generate_seq(start, learner, length):
    seq = start
    while len(seq.split(' ')) < length:
        seq = learner.predict(seq)
    return seq


def convert_output(seq):
    tokens = seq.replace(',', ' ').split(' ')
    prev_tok = None

    cleaned = []
    for t in tokens:
        if prev_tok and prev_tok[0] == 'H' and t[0] == 'H':
            if prev_tok.split('-')[1] != t.split('-')[1]:
                t = t.replace('H', 'S')
        cleaned.append(t)
        prev_tok = t
    return cleaned


def run(args):
    model_dir, fname = os.path.split(args.model)

    learn = load_learner(model_dir, fname)

    enc_seq = generate_seq(args.prompt, learn, args.seq)
    tokens = convert_output(enc_seq) * args.loop

    dec_seq = Track.decode_notes(tokens)
    stream = music21.stream.Stream()
    t = music21.tempo.MetronomeMark(number=args.tempo)
    stream.append(t)
    for note in dec_seq:
        stream.append(note)

    sp = music21.midi.realtime.StreamPlayer(stream)
    sp.play()
