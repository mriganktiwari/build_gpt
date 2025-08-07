import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class TextLoader:
    def __init__(self, vocab, stoi, itos, train_data, val_data):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.stoi = stoi
        self.itos = itos
        self.train_data = train_data
        self.val_data = val_data
    
    @classmethod
    def from_text(cls, text, train_split=0.9):
        vocab = sorted(list(set(text)))
        stoi = {ch:i for i,ch in enumerate(vocab)}
        itos = {i:ch for ch,i in stoi.items()}

        data = [stoi[ch] for ch in text]
        n = int(train_split * len(data))
        train_data = data[:n]
        val_data = data[n:]
        return cls(vocab, stoi, itos, train_data, val_data)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos[token] for token in tokens])

    def get_batch(self, split, batch_size=8, block_size=32):
        self.block_size = block_size
        split_data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(0, len(split_data) - block_size, (batch_size,))
        x = [split_data[i : i + block_size] for i in ix]
        y = [split_data[i+1 : i + block_size + 1] for i in ix]
        return torch.tensor(x), torch.tensor(y)

# evaluate loss
@torch.no_grad()
def estimate_loss(model, data_loader, device, eval_iters=250):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data_loader.get_batch(split, batch_size=128, block_size=128)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, data_loader.vocab_size), y.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # forget, input, cell state, output:
        # all gates weights stacked horizontally
        self.wx_gates = nn.Parameter(torch.randn(self.input_size,  self.hidden_size * 4) * 0.01)
        self.wh_gates = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size * 4) * 0.01)
        self.b_gates  = nn.Parameter(torch.zeros(self.hidden_size * 4))

    def forward(self, x, hidden):
        # x: (b, self.input_size)
        # B,T = x.shape
        h_prev, c_prev = hidden

        x_gates = x @ self.wx_gates             # (b,n) @ (n,h) -> (b,h)
        h_prev_gates = h_prev @ self.wh_gates   # (b,h) @ (h,h) -> (b,h)
        gates_out = x_gates + h_prev_gates + self.b_gates
        ft, it, gt, ot = gates_out.chunk(4, dim=1)

        ft = torch.sigmoid(ft)
        it = torch.sigmoid(it)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)

        c_t = (ft * c_prev) + (it * gt)
        h_t = torch.tanh(c_t) * ot

        return h_t, c_t
    
class MultiLayerLSTM(nn.Module):
    def __init__(self, vocab_size, n_layers, input_size, hidden_size, dropout = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = n_layers

        self.emb_layer = nn.Parameter(torch.randn(self.vocab_size, self.input_size) * 0.02)
        self.lstm_cells = nn.ModuleList()

        # first layer
        self.lstm_cells.append(LSTMCell(self.input_size, self.hidden_size))
        # additional layers
        for layer in range(1, self.num_layers):
            self.lstm_cells.append(LSTMCell(self.hidden_size, self.hidden_size))
        
        self.dropout_layer = nn.Dropout(self.dropout) if dropout > 0 and n_layers > 1 else None

        self.why = nn.Parameter(torch.randn(self.hidden_size, self.vocab_size))
        self.by = nn.Parameter(torch.zeros(self.vocab_size))
    
    def forward(self, x, hidden=(None,None)):
        # x: (b,t)
        # hidden: tuple of (h_prev, c_prev)
        B,T = x.shape
        hs, cs = hidden
        if hs is None and cs is None:
            hs = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
            cs = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)

        logits = []
        x_emb = self.emb_layer[x] # (b,t,n)
        for t in range(T):
            # We'll build new tensors for the states
            new_hs = torch.zeros_like(hs, device=x.device)
            new_cs = torch.zeros_like(cs, device=x.device)

            xt = x_emb[:, t, :] # (b,n)
            for layer in range(self.num_layers):
                h_prev_layer = hs[layer]
                c_prev_layer = cs[layer]

                cell_layer = self.lstm_cells[layer]
                h_new, c_new = cell_layer(xt, (h_prev_layer, c_prev_layer))
                
                new_hs[layer] = h_new
                new_cs[layer] = c_new
                # Apply dropout between layers (not after last layer)
                if layer < self.num_layers - 1 and self.dropout_layer is not None:
                    xt = self.dropout_layer(h_new)
                else:
                    xt = h_new
            
            # Update for next timestep
            hs, cs = new_hs, new_cs
            yt = hs[-1] @ self.why + self.by
            logits.append(yt) # T elements of shape (b,vocab_size)
        logits = torch.stack(logits, dim=1) # (B,T,vocab_size)
        return logits