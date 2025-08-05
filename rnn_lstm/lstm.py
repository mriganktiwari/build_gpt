import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

#-------------------------------------------
# data, vocab, mappings, splits
words = open('../makemore/names.txt', 'r').read().splitlines()
vocab = sorted(list(set(''.join(words))))
stoi = {ch:i+1 for i,ch in enumerate(vocab)}
stoi['.'] = 0
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(itos)

n = int(len(words) * 0.9)
train_words = words[:n]
val_words = words[n:]

# hparams
batch_size = 64
n_embd = 64
hidden_dim = 128
max_iters = 20000
n_layers = 2
eval_iters = 250
eval_interval = 1000

# get batch of data
def get_batch(split, bs = batch_size):
    inp_seq, out_seq = [], [] # list of sequence of chars in input and output (shifted by 1)
    data = train_words if split == 'train' else val_words
    ix = torch.randint(len(data)-1, (batch_size,))
    batch = [data[i] for i in ix]
    for word in batch:
        chs = word + '.'
        inp, out = [], []
        inp = [stoi[ch1] for ch1,_ in zip(chs, chs[1:])]
        out = [stoi[ch2] for _,ch2 in zip(chs, chs[1:])]

        inp_seq.append(torch.tensor(inp))
        out_seq.append(torch.tensor(out))
    inp_seq_padded = pad_sequence(inp_seq, batch_first=True, padding_value=0)
    out_seq_padded = pad_sequence(out_seq, batch_first=True, padding_value=-1)
    return inp_seq_padded, out_seq_padded

# evaluate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    # model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # model.train()
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
    def __init__(self, vocab_size, n_layers, input_size, hidden_size, dropout = 0.2, ):
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
        x = self.emb_layer[x] # (b,t,n)
        for t in range(T):
            xt = x[:, t, :] # (b,n)
            for layer in range(self.num_layers):
                h_prev_layer = hs[layer]
                c_prev_layer = cs[layer]

                h_new, c_new = self.lstm_cells[layer](xt, (h_prev_layer, c_prev_layer))
                hs[layer] = h_new
                cs[layer] = c_new
                # Apply dropout between layers (not after last layer)
                if layer < self.num_layers - 1 and self.dropout_layer is not None:
                    xt = self.dropout_layer(h_new)
                else:
                    xt = h_new
            yt = hs[-1] @ self.why + self.by
            logits.append(yt) # T elements of shape (b,vocab_size)
        logits = torch.stack(logits, dim=1) # (B,T,vocab_size)
        return logits

model = MultiLayerLSTM(vocab_size, n_layers=n_layers, input_size=n_embd, hidden_size=hidden_dim)
print(f'{sum(p.numel() for p in model.parameters())} parameters')
                
# forward pass
xb, yb = get_batch('train')
logits = model(xb)
print(f'logits shape: {logits.shape}')

loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), yb.view(-1), ignore_index=-1)
print(f'loss = {loss.item()}')





















































# -------------------------------------------------------------------------------------------
# class SimpleLSTM:
#     def __init__(self, n_embd=n_embd, hidden_dim=hidden_dim):
#         self.n_embd = n_embd
#         self.hidden_dim = hidden_dim

#         self.emb_layer = torch.randn((vocab_size, n_embd)) * 0.1

#         # forget gate
#         self.wxf = torch.randn(n_embd, hidden_dim) * 0.01
#         self.whf = torch.randn(hidden_dim, hidden_dim) * 0.01
#         self.bf  = torch.zeros(hidden_dim)

#         # input gate
#         self.wxi = torch.randn(n_embd, hidden_dim) * 0.01
#         self.whi = torch.randn(hidden_dim, hidden_dim) * 0.01
#         self.bi  = torch.zeros(hidden_dim)

#         # cell candidate gate
#         self.wxg = torch.randn(n_embd, hidden_dim) * 0.01
#         self.whg = torch.randn(hidden_dim, hidden_dim) * 0.01
#         self.bg  = torch.zeros(hidden_dim)

#         # output gate
#         self.wxo = torch.randn(n_embd, hidden_dim) * 0.01
#         self.who = torch.randn(hidden_dim, hidden_dim) * 0.01
#         self.bo  = torch.zeros(hidden_dim)

#         self.why = torch.randn(hidden_dim, vocab_size) * 0.01
#         self.by  = torch.zeros(vocab_size)

#         self.parameters = [
#             self.emb_layer,
#             self.wxf, self.whf, self.bf,
#             self.wxi, self.whi, self.bi,
#             self.wxg, self.whg, self.bg,
#             self.wxo, self.who, self.bo,
#             self.why, self.by
#         ]

#         for p in self.parameters:
#             p.requires_grad = True

#     def __call__(self, x):
#         # x: (b,t)
#         B,T = x.shape
#         h = torch.zeros(B, self.hidden_dim) # hidden state (b,h)
#         c = torch.zeros(B, hidden_dim) # cell state (b,h)

#         x_emb = self.emb_layer[x] # (b,t,n_embd)
#         logits = []
#         for t in range(T):
#             xt = x_emb[:, t, :] # (b,n_embd)
#             ft = torch.sigmoid(xt @ self.wxf + h @ self.whf + self.bf) # (b,h)
#             it = torch.sigmoid(xt @ self.wxi + h @ self.whi + self.bi) # (b,h)
#             gt = torch.tanh(   xt @ self.wxg + h @ self.whg + self.bg) # (b,h)
#             ot = torch.sigmoid(xt @ self.wxo + h @ self.who + self.bo) # (b,h)

#             c = (ft * c) + (it * gt)
#             h = ot * torch.tanh(c)

#             yt = h @ self.why + self.by # (b,vocab_size)
#             logits.append(yt) # list of T elements of shape (b,vocab_size)
#         return torch.stack(logits, dim=1) # (b,T,vocab_size)
# -------------------------------------------------------------------------------------------