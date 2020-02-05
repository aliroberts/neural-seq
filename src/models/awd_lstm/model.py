import sys

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F

from tqdm import tqdm

from .locked_dropout import LockedDropout
from .weight_drop import WeightDrop


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
            1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(
            masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse
                                      )
    return X


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp=300, nhid=600, nlayers=4, rnn_type='LSTM', dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0.0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (
                ninp if tie_weights else nhid), 1, dropout=0, batch_first=False) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(
                    rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l !=
                                      nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(
                    rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (
                ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        for rnn in self.rnns:
            if isinstance(rnn, nn.LSTM):
                rnn.flatten_parameters()

    def reset(self):
        if self.rnn_type == 'QRNN':
            [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input,
                               dropout=self.dropoute if self.training else 0)
        # emb = self.idrop(emb)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        decoded = self.decoder(output.view(
            output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))

        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                     weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

    def predict(self, prompt):
        """
        Generate a raw output vector
        """
        self.hidden = self.init_hidden(1)
        output, hidden = self(prompt, self.hidden)
        self.hidden = hidden
        return output


def train(data, model, args, lr=1e-3, t0=0, lambd=0, wdecay=1.2e-6, alpha=0, beta=0, clip=0.25, when='0', optim='adam', gpu=False):
    dest = args.dest
    epochs = args.epochs
    criterion = nn.CrossEntropyLoss()

    if optim.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wdecay)
    elif optim.lower() == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr,
                                     t0=t0, lambd=lambd, weight_decay=wdecay)
    else:
        print('Unrecognized optimizer')
        sys.exit(1)
    train_dl = data.train_dl
    valid_dl = data.valid_dl
    vocab_sz = len(data.vocab.itos)
    if gpu:
        model.cuda()
    hidden = model.init_hidden(data.bs)

    when = map(lambda x: int(x), when.split(','))
    model.train()

    for epoch in range(1, epochs + 1):
        if epoch in when:
            lr /= 10
            print(f'Learning rate reduced ({lr})')
        print('Training...ls')
        for xb, yb in tqdm(train_dl):
            # xb and yb here will contain indices for a lookup table for each symbol in our vocabulary
            # They will be of dimension bs * bptt

            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()

            # Note that since this LSTM doesn't use batch_first structure we need
            # to transpose the input to be of shape (bptt * bs)

            output, hidden, rnn_hs, dropped_rnn_hs = model(
                xb.t(), hidden, return_h=True)

            flat_yb = yb.view(-1)
            raw_loss = criterion(output.view(-1, vocab_sz), flat_yb)

            loss = raw_loss
            # Activiation Regularization
            loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean()
                              for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + \
                sum(beta * (rnn_h[1:] - rnn_h[:-1]
                            ).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        if args.save_freq and epoch % args.save_freq == 0 or epoch == epochs:
            checkpoint_name = f'model-epoch-{epoch}.pth'
            print(f'Saving checkpoint {checkpoint_name}')
            torch.save(model.state_dict(),
                       f'{args.dest}/{checkpoint_name}')
        model.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(model(xb.t(), model.init_hidden(
                data.bs))[0].view(-1, vocab_sz), yb.view(-1)) for xb, yb in valid_dl) / len(valid_dl)
        print(f'{epoch} Train: {raw_loss.data} Valid: {valid_loss}')
