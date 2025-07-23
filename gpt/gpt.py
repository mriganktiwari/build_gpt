import torch
import torch.nn as nn
import torch.nn.functional as F

# hparams
torch.manual_seed(2)

block_size = 256
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 50000
learning_rate = 3e-4
eval_iters = 250
eval_interval = 1000
n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.2

# data
shakespeare = open('input.txt', 'r').read()
vocab = sorted(list(set(''.join(shakespeare))))
vocab_size = len(vocab)
stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for ch,i in stoi.items()}

encode = lambda text: [stoi[ch] for ch in text]
decode = lambda idx: ''.join([itos[i] for i in idx])
encoded_text = encode(shakespeare)

n = int(len(encoded_text) * 0.9)
train_data = encoded_text[:n]
val_data = encoded_text[n:]

# evaluate loss
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

# data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(train_data) - block_size, (batch_size,))

    x = torch.tensor([train_data[i : i + block_size] for i in ix])
    y = torch.tensor([train_data[i+1 : i + 1 + block_size] for i in ix])

    return x.to(device), y.to(device)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (b, t, n_embd) @ (n_embd, head_size) --> (b, t, head_size)
        q = self.query(x) # (b, t, n_embd) @ (n_embd, head_size) --> (b, t, head_size)
        v = self.value(x) # (b, t, n_embd) @ (n_embd, head_size) --> (b, t, head_size)
        wei = q @ k.transpose(-2, -1) / (C**0.5) # (b, t, head_size) @ (b, head_size, t) --> (b, t, t)
        wei = wei.masked_fill_(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (b, t, t)
        wei = self.dropout(wei)

        out = wei @ v # (b, t, t) @ (b, t, head_size) --> (b, t, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # projection layer after sa heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (b, t, head_size*num_heads)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer after ffwd
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x +     self.ffwd(self.ln2(x))
        return x

# model class
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        blocks = [Block(n_embd, n_heads) for _ in range(n_layers)]
        self.blocks = nn.Sequential(*blocks)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets = None):
        # x shape       - (b, t)
        # targets shape - (b, t)
        B, T = x.shape

        tok_emb = self.token_embedding_table(x) # (b, t, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (t, n_embd)
        tok_emb += pos_emb # (b, t, n_embd)
        x = self.blocks(tok_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (b, t, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # B,T,C = logits.shape
            loss = F.cross_entropy(logits.transpose(-1,-2), targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (b, t)
        for _ in range(max_new_tokens):
            idx_chopped = idx[:, -block_size:]
            logits, loss = self(idx_chopped) # (b, t, vocab_size)
            logits = logits[:, -1, :] # (b, vocab_size)
            probs = F.softmax(logits, dim=-1) # (b, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (b, t+1)
        return idx

# training
model = GPT().to(device)
print(f'{sum([p.numel() for p in model.parameters()])} parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generation
idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))