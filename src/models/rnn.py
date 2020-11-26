import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm


class RNN(nn.Module):
    """
    A simple RNN comprising of an embedding layer, a configurable number of LSTM layers and a linear decoder
    """

    def __init__(self, vocab_sz, emb_sz=300, n_hid=600, n_layers=1):
        super().__init__()
        self.encoder = nn.Embedding(vocab_sz, emb_sz)
        self.lstm = nn.LSTM(input_size=emb_sz,
                            hidden_size=n_hid, num_layers=n_layers)
        self.decoder = nn.Linear(n_hid, vocab_sz)

    def forward(self, mb):
        x = self.encoder(mb)
        x, (h, c) = self.lstm(x)
        return self.decoder(x), h, c

    def predict(self, seq):
        # The input seq to the model are of dimensions (seq, batchsize)
        seq = seq.unsqueeze(-1)
        out, _, _ = self(seq)
        return out


def train(data, model, epochs, dest, lr=3e-4):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    loss_func = F.cross_entropy
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        epoch_loss = 0
        num = 0
        for xb, yb in tqdm(data.train_dl):
            x = xb.to(device).t()
            y = yb.to(device).t()
            model.zero_grad()
            out, _, _ = model(xb.t())
            loss = F.cross_entropy(
                out.reshape(-1, len(data.vocab.itos)), y.reshape(-1))
            epoch_loss += loss.item()
            num += 1
            loss.backward()
            optim.step()
        print(f'epoch {i} loss {epoch_loss/num}')
        if (i + 1) % 25 == 0:
            torch.save(model.state_dict(), dest/f'model-{i + 1}.pth')
    torch.save(model.state_dict(), dest/'model.pth')
