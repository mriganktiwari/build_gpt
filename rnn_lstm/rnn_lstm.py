import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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
n_embd = 128
hidden_dim = 128
max_iters = 10000
n_layers = 3
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

# model class
class MultiLayerRNN:
    def __init__(self, n_embd=n_embd, hidden_dim=hidden_dim, n_layers=n_layers):
        self.n_embd = n_embd
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.emb_layer = torch.randn((vocab_size, n_embd)) * 0.1 # (27, n_embd)

        self.wxh = [torch.randn(n_embd if i==0 else hidden_dim, hidden_dim) * 0.01 for i in range(n_layers)] # (b,t,n_embd) -> (b,t,hidden_dim)
        self.whh = [torch.randn(hidden_dim, hidden_dim) * 0.01 for _ in range(n_layers)] # (b,t,hidden_dim) -> (b,t,hidden_dim)
        self.bh  = [torch.zeros(hidden_dim) for _ in range(n_layers)]

        self.why = torch.randn(hidden_dim, vocab_size) * 0.01 # (b,t,hidden_dim) -> (b,t,vocab_size)
        self.by  = torch.zeros(vocab_size)

        self.parameters = [self.emb_layer , self.why, self.by]
        for layer in range(n_layers):
            self.parameters += [self.wxh[layer], self.whh[layer], self.bh[layer]]

        for p in self.parameters:
            p.requires_grad = True

    def __call__(self, x):
        # x: (b,t)
        B,T = x.shape
        hs = [torch.zeros(B, self.hidden_dim) for _ in range(self.n_layers)] # list of h for each layer in n_layers
        x_emb = self.emb_layer[x]
        logits = []
        
        for t in range(T):
            xt = x_emb[:, t, :] # (b, n_embd)

            for layer in range(n_layers):
                h_prev = hs[layer]
                hs[layer] = torch.tanh(xt @ self.wxh[layer] + h_prev @ self.whh[layer] + self.bh[layer]) # (b, hidden_dim)
                # (b, n_embd) @ (n_embd, hidden_dim) --> (b, hidden_dim) + (b, hidden_dim) @ (hidden_dim, hidden_dim) --> (b, hidden_dim)
                xt = hs[layer] # (b, hidden_dim)
            yt = hs[-1] @ self.why + self.by # (b, vocab_size)
            logits.append(yt)

        logits = torch.stack(logits, dim=1) # (b, T, vocab_size)
        return logits

# model init
model = MultiLayerRNN()
print(f'{sum(p.numel() for p in model.parameters)} params')

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1), ignore_index=-1)

    for p in model.parameters:
        p.grad = None

    loss.backward()
    if iter % eval_interval == 0:
        losses = estimate_loss()
        total_norm = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters if p.grad is not None))
        print(f'Iteration {iter} train loss = {losses['train']:.4f} | val loss = {losses['val']:.4f} | Gradient norm: {total_norm:.4f}')

    lr = 0.1 if iter < 4000 else 0.01
    for p in model.parameters:
        p.data += -(lr * p.grad)

# generate
print(f'\nStarting generation')
print('--'*10)
for _ in range(5):
    idx = torch.zeros(1,1, dtype=torch.long)
    h = torch.zeros(1, hidden_dim)
    # emb = emb_layer[idx]
    # print(f'emb shape: {emb.shape}')
    count = 0
    while True:
        logits = model(idx)[:,-1]
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        idx = torch.cat([idx, torch.tensor([[next_idx]])], dim=1)
        if next_idx == 0:
            break
    print(''.join([itos[i.item()] for i in idx[0][1:]]))