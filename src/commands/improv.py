
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

from collections import OrderedDict

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


def predict_beam(prompt, model, encoder, vocab, args):
    raise NotImplementedError


def predict_topk(prompt, model, encoder, vocab, args):
    raise NotImplementedError


def predict_nucleus(prompt, model, encoder, vocab, args):
    model.eval()
    vocab_sz = len(vocab.itos)

    # Store the generated string sequence
    generated = prompt

    with torch.no_grad():
        while encoder.duration(generated) < args.seq:
            out = model.predict(torch.LongTensor(
                vocab.numericalize(generated)))

            # Get the last vector of logprobs
            logprobs = out.reshape(-1, vocab_sz)[-1]

            nucleus_probs = []
            nucleus_indices = []

            sorted_probs = F.softmax(logprobs/1, -1).sort(descending=True)
            for p, idx in zip(sorted_probs[0], sorted_probs[1]):
                nucleus_probs.append(p)
                nucleus_indices.append(idx)
                if sum(nucleus_probs) > 0.9:
                    break

            unnormalised = torch.Tensor(nucleus_probs)
            probs = unnormalised * (1/sum(torch.Tensor(unnormalised)))
            # We need to refer back to the original indices to grab the correct vocab elements
            prediction = nucleus_indices[torch.distributions.Categorical(
                probs).sample().item()]
            generated.append(vocab.itos[prediction])
    return generated


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

    state_dict = torch.load(args.state, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove 'module.' of dataparallel
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)

    # model.load_state_dict(torch.load(
    #     args.state, map_location=torch.device('cpu')))

    predict_func = PREDICT_CHOICES[args.sample]

    # If no prompt has been provided, choose a random token from our vocab to start with
    prompt = args.prompt.split(',') if args.prompt else [
        random.choice(vocab.itos)]

    enc_seq = predict_func(prompt, model, encoder, vocab, args)

    # enc_seq = generate_seq(args.prompt, learn, args.seq)
    tokens = enc_seq * args.loop

    decoded = encoder.decode(tokens, tempo=args.tempo)
    play_midi(decoded)
