import torch
import torch.nn as nn
import torch.nn.functional as F

# hparams
torch.manual_seed(2)

block_size = 8
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 10000
learning_rate = 1e-3

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

# data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(train_data) - block_size, (batch_size,))

    x = torch.tensor([train_data[i : i + block_size] for i in ix])
    y = torch.tensor([train_data[i+1 : i + 1 + block_size] for i in ix])

    return x.to(device), y.to(device)

# model class
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_encoding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, x, targets = None):
        # x shape       - (b, t)
        # targets shape - (b, t)
        logits = self.token_encoding_table(x) # (b, t, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # B,T,C = logits.shape
            loss = F.cross_entropy(logits.transpose(-1,-2), targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (b, t)
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # (b, t, vocab_size)
            logits = logits[:, -1, :] # (b, vocab_size)
            probs = F.softmax(logits, dim=-1) # (b, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (b, t+1)
        return idx

# training
model = BigramLM(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# generation
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))