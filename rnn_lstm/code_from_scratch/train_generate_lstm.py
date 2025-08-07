import time
import torch.nn.functional as F
import torch
from rnn_lstm.lstm import MultiLayerLSTM, estimate_loss, TextLoader

text = open('gpt/input.txt', 'r').read()
loader = TextLoader.from_text(text)

device = 'cuda'
model = MultiLayerLSTM(loader.vocab_size, n_layers=3, input_size=512, hidden_size=512, dropout=0.5)
params = model.parameters()
print(f'{sum(p.numel() for p in params)} parameters')
model = model.to(device)
for p in params:
    p.requires_grad = True

max_iters = 20000
batch_size = 128
block_size = 128
eval_interval = 1000

for iter in range(max_iters):
        xb, yb = loader.get_batch('train', batch_size=batch_size, block_size=block_size)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)

        loss = F.cross_entropy(logits.view(-1, loader.vocab_size), yb.view(-1), ignore_index=-1)

        # zero the gradients
        for p in model.parameters():
            p.grad = None

        loss.backward()
        
        # print loss
        if iter % eval_interval == 0:
            losses = estimate_loss(model, loader, device)
            total_norm = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))
            print(f'Iteration {iter} train loss = {losses["train"]:.4f} | val loss = {losses["val"]:.4f} | Gradient norm: {total_norm:.4f}')
        
        # weights update
        lr = 1e-1 if iter < 20000 else 1e-2
        for p in model.parameters():
            p.data += -(lr * p.grad)

def generate(model, loader, prompt=None, device=device, max_new_tokens=200):
    with torch.no_grad():
        model.eval()
        if not prompt:
            idx = torch.tensor([[0]], dtype=torch.long, device=device)
        # else:
        #     prompt_tokens = loader.encode(prompt)
        #     idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        batch_size = 1
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            if idx.shape[1] > loader.block_size:
                idx = idx[:, -loader.block_size:] # (b,T)
            
            logits = model(idx) # (b,T,vocab_size)
            logits = logits[0][-1] # (vocab_size,) vector from last time step
            probs = F.softmax(logits, dim=-1)
            
            next_idx = torch.multinomial(probs, 1).item()
            idx = torch.cat([idx, torch.tensor([[next_idx]], device=device)], dim=1)
        
        print(''.join([loader.itos[i.item()] for i in idx[0][1:]]))
        model.train()
        
generate(model, loader)