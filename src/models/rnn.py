import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.manual_seed(42)


class RNN(nn.Module):
    """
    A simple RNN comprising of an embedding layer, a configurable number of LSTM layers and a linear decoder
    """

    def __init__(self, vocab_sz, emb_sz=300, n_hid=600, n_layers=1):
        super(RNN, self).__init__()
        self.encoder = nn.Embedding(vocab_sz, emb_sz)
        self.lstm = nn.LSTM(input_size=emb_sz,
                            hidden_size=n_hid, num_layers=n_layers)
        self.decoder = nn.Linear(n_hid, vocab_sz)

    def forward(self, x, hidden):
        emb = self.encoder(x)
        # output contains the history of hidden states; hidden is a tuple of
        # (hidden_state, cell_state)
        output, hidden = self.lstm(self.encoder(x), hidden)
        decoded = self.decoder(output)
        # Decoded will be of dim bs * bptt * vocab_sz
        return decoded, hidden


def train(data, model, epochs, dest, lr=0.1):
    loss_func = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_dl = data.train_dl
    for epoch in range(epochs):
        # Not really needed here yet since we have no BatchNorm or DropOut layers but set anyway
        model.train()
        for xb, yb in train_dl:
            # xb and yb here will contain indices for a lookup table for each symbol in our vocabulary
            # They will be of dimension bs * bptt
            output, hidden = model(xb, None)
            vocab_sz = len(data.vocab.itos)
            # Flatten our outputs (get rid of first batch dimension)
            flat_output = output.view(-1, vocab_sz)
            flat_yb = yb.view(-1)
            loss = loss_func(flat_output, flat_yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Calculate training loss
        with torch.no_grad():
            train_loss = sum(loss_func(
                model(xb, None)[0].view(-1, vocab_sz), yb.view(-1)) for xb, yb in train_dl) / len(train_dl)
        print(epoch + 1, train_loss)

        # # Calculate validation loss
        # with torch.no_grad():
        #     valid_dl = data.valid_dl
        #     valid_loss = sum(
        #         loss_func(model(xb)[0].view(-1, vocab_sz), yb) for xb, yb in valid_dl) / len(valid_dl)
        # print(epoch, valid_loss)
