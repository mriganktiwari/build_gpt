import torch
import torch.nn as nn
import torch.nn.functional as F

# data loader -------------------------------------------------------------------------------------------------------
class TextLoader:
    def __init__(self, text, test_split=0.9):
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = {i:ch for ch,i in self.stoi.items()}

        self.data = [self.stoi[ch] for ch in text]
        n = int(test_split * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        
    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos[token] for token in tokens])

    def get_batch(self, split, batch_size, block_size, device='cpu'):
        data = self.train_data if split=='train' else self.val_data
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = [data[i : i + block_size] for i in ix]
        y = [data[i+1 : i+1 + block_size] for i in ix]
        return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

# evaluate loss -------------------------------------------------------------------------------------------------------
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

# model -------------------------------------------------------------------------------------------------------
class LSTMCell(nn.Module):
    def __init__(self, input_embd, hidden_embd):
        super().__init__()
        # weights: input -> 4*hidden, hidden -> 4*hidden
        self.wx_gates = nn.Parameter(torch.empty(input_embd, hidden_embd * 4))
        self.wh_gates = nn.Parameter(torch.empty(hidden_embd, hidden_embd * 4))
        self.bx_gates = nn.Parameter(torch.zeros(hidden_embd * 4))
        self.bh_gates = nn.Parameter(torch.zeros(hidden_embd * 4))

        # better init
        nn.init.xavier_normal_(self.wx_gates)
        nn.init.orthogonal_(self.wh_gates)
        # set forget gate bias positive (first quarter is forget gate because we chunk as ft,it,gt,ot)
        hidden = hidden_embd
        # both biases help
        with torch.no_grad():
            self.bx_gates[:hidden] += 1.0
            self.bh_gates[:hidden] += 1.0

    def forward(self, x, h_prev, c_prev):
        # x: (B, input_embd)
        x_gates = x @ self.wx_gates + self.bx_gates
        h_gates = h_prev @ self.wh_gates + self.bh_gates
        gates_output = x_gates + h_gates

        ft, it, gt, ot = gates_output.chunk(4, dim=1)
        ft = torch.sigmoid(ft)
        it = torch.sigmoid(it)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)

        c_t = ft * c_prev + it * gt
        h_t = ot * torch.tanh(c_t)
        return h_t, c_t


class MultiLayerLSTM(nn.Module):
    def __init__(self, vocab_size, input_embd, hidden_embd, layers, dropout=0.5, tie_weights=True):
        super().__init__()
        self.layers = layers
        self.hidden_embd = hidden_embd

        # use nn.Embedding
        self.embedding = nn.Embedding(vocab_size, input_embd)

        # build layers
        self.lstm_layer = nn.ModuleList()
        self.lstm_layer.append(LSTMCell(input_embd, hidden_embd))
        for _ in range(1, layers):
            self.lstm_layer.append(LSTMCell(hidden_embd, hidden_embd))

        self.dropout = nn.Dropout(dropout) if dropout is not None and layers > 1 else None

        # decoder bias
        self.by = nn.Parameter(torch.zeros(vocab_size))
        self.tie_weights = tie_weights and (input_embd == hidden_embd)
        # if tie_weights and dims match, we'll use embedding weights transposed in forward pass (no extra parameter)

    def forward(self, x):
        B,T = x.shape
        device = x.device
        hs = torch.zeros(self.layers, B, self.hidden_embd, device=device)
        cs = torch.zeros(self.layers, B, self.hidden_embd, device=device)
        logits = []

        emb = self.embedding(x)  # (B,T,input_embd)

        for t in range(T):
            xt = emb[:, t, :]  # (B, input_embd)
            hs_new = torch.zeros_like(hs, device=device)
            cs_new = torch.zeros_like(cs, device=device)
            for layer in range(self.layers):
                h_layer, c_layer = hs[layer], cs[layer]
                cell_layer = self.lstm_layer[layer]
                h_new, c_new = cell_layer(xt, h_layer, c_layer)
                hs_new[layer] = h_new
                cs_new[layer] = c_new
                if layer < self.layers - 1 and self.dropout is not None:
                    xt = self.dropout(h_new)
                else:
                    xt = h_new
            hs = hs_new
            cs = cs_new

            top_h = hs[-1]  # (B, hidden_embd)
            if self.tie_weights:
                # use embedding weights transposed as decoder weights
                # embedding.weight: (vocab, emb) -> transpose -> (emb, vocab)
                yt = top_h @ self.embedding.weight.t() + self.by
            else:
                # fallback: create a linear projection on the fly (less efficient)
                # create a weight on the fly from param memory is not ideal — recommend tie_weights=True when possible
                if not hasattr(self, 'why'):
                    self.why = nn.Parameter(torch.empty(self.hidden_embd, self.embedding.num_embeddings))
                    nn.init.xavier_normal_(self.why)
                yt = top_h @ self.why + self.by
            logits.append(yt)

        logits = torch.stack(logits, dim=1)  # (B, T, vocab)
        return logits
    
@torch.no_grad()
def generate(model, stoi, itos, block_size, prompt=None, device='cuda', max_new_tokens=500, out_path=None):
    model.eval()
    # Build batch shape (1, T)
    if prompt is None or len(prompt) == 0:
        idx = torch.tensor([[0]], dtype=torch.long, device=device)
    else:
        seq = [stoi[ch] for ch in prompt]
        idx = torch.tensor([seq], dtype=torch.long, device=device)  # shape (1, T)
    generated_chars = []

    for _ in range(max_new_tokens):
        idx_cropped = idx[:, -block_size:]  # (1, Tc)
        logits = model(idx_cropped)  # (1, Tc, vocab)
        logits = logits[0, -1, :]    # last token logits, shape (vocab,)
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        generated_chars.append(itos[next_idx])
        # append to idx for next step
        next_idx_tensor = torch.tensor([[next_idx]], device=device, dtype=torch.long)
        idx = torch.cat([idx, next_idx_tensor], dim=1)

    full_text = ''.join(generated_chars)
    # if out_path:
    #     with open(out_path, 'w', encoding='utf-8') as fp:
    #         fp.write(full_text)
    # print(full_text)
    model.train()
    return full_text