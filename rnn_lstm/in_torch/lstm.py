import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, emb=384, hidden=384, layers=2, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)
    
    def forward(self, x):
        """
        x: (b,t) shape
        returns:
        logits: (b,t,vocab_size)
        hidden_state: (layers, hidden)
        """
        x = self.emb(x)
        out, hc = self.lstm(x)
        logits = self.head(out)
        return logits, hc

@torch.no_grad()
def generate(model, stoi, itos, block_size, prompt=None, device='cpu', max_new_tokens=500):
    model.eval()
    
    if not prompt:
        idx = torch.tensor([[0]], dtype=torch.long, device=device)
    else:
        idx = torch.tensor([stoi[ch] for ch in prompt], dtype=torch.long, device=device)
    batch_size = idx.shape[0]
    generated_tokens = []
    for _ in range(max_new_tokens):
        idx_cropped = idx[:, -block_size:] # (b,T)
        logits,_ = model(idx_cropped) # (b,T,vocab_size)
        logits = logits[0][-1] # (vocab_size,) vector from last time step
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        idx = torch.cat([idx, torch.tensor([[next_idx]], device=device)], dim=1)
    print(''.join([itos[i.item()] for i in idx[0][1:]]))
    
    model.train()