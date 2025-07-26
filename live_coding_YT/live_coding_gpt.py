import torch
import torch.nn as nn
import torch.nn.functional as F

# hparams ---------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 8
batch_size = 4
eval_iters = 250
max_iters = 10000
eval_interval = 1000
n_embd = 32

torch.manual_seed(2)

# reading data ---------------------------------------------------------------------
shakespeare = open('../gpt/input.txt', 'r').read()
vocab = sorted(list(set(''.join(shakespeare))))
vocab_size = len(vocab)
stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for ch,i in stoi.items()}

# encode list of characters to list of integers
encode = lambda s: [stoi[ch] for ch in s]
# decode list of int to list of chars
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(shakespeare))
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

# data loader ---------------------------------------------------------------------
def get_batch(split):
    d = train_data if split=='train' else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    xb = torch.stack([d[i   : i +   block_size] for i in ix], dim=0)
    yb = torch.stack([d[i+1 : i+1 + block_size] for i in ix], dim=0)
    return xb.to(device), yb.to(device)

# estimate loss -------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# mdoel class ---------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x) # (b,t,head_size)
        k = self.key(x) # (b,t,head_size)
        v = self.value(x) # (b,t,head_size)

        wei = q @ k.transpose(-2,-1) / (self.head_size ** 0.5) # (b,t,head_size) @ (b,head_size,t) --> (b,t,t)
        wei = wei.masked_fill_(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (b,t,t)

        out = wei @ v # (b,t,t) @ (b,t,head_size) --> (b,t,head_size)
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # list of (b,t,head_size//n_heads) --> (b,t,head_size)
        out = self.proj(out) # (b,t,n_embd)
        return out

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadedAttention(4, n_embd//4)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets = None):
        B,T = x.shape

        tok_emb = self.token_embedding_table(x) # (b, t, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (b, t, n_embd)
        tok_emb += pos_emb
        x = self.sa_heads(tok_emb)
        logits = self.lm_head(x) # (b, t, vocab_size)

        if targets is None:
            loss = None
        else:
            # targets: (b, t)                                   --> (b*t)
            # logits: (b, t, c) - c: channels --> (b, c, t)     --> (b*t, c)
            # loss = F.cross_entropy(logits.transpose(-1,-2), targets)

            B,T,C = logits.shape
            logits_new = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits_new, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens=100):
        # idx: (b, t)
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -block_size:] # (b, block_size)
            logits, _ = self(idx_cropped) # (b, vocab_size)
            logits = logits[:, -1, :] # (b, vocab_size)
            probs = F.softmax(logits, dim=-1) # (b, vocab_size)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_new), dim=-1) # (b, t+1)
        return idx

# model init ---------------------------------------------------------------------
model = BigramLM()
print(f'{sum(p.numel() for p in model.parameters())} parameters')
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)

# training -----------------------------------------------------------------------
for iter in range(max_iters):
    # lr = 1e-3 if iter < 10000 else 1e-4
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    
    # get a batch
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)

    if iter % eval_interval == 0:
        out = estimate_loss()
        print(f'Iteration {iter + 1} : Train Loss = {out['train']:.4f}, Validation Loss = {out['val']:.4f}')

    # set grad to None and backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generation ---------------------------------------------------------------------
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx)[0].tolist()))
