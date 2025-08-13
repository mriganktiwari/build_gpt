import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    block_size: int = 1024

config = GPTConfig()

# model ----------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        head_size = config.n_embd / config.n_head
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_head * head_size) # config.n_embd = n_head * head_size
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size, config.block_size), # (1, 1, T, T)
                             persistent=False)

    def forward(self, x):
        B,T,C = x.shape # (b,t,n_embd)
        qkv = self.c_attn(x) # (b,t, 3*n_embd)
        q,k,v = qkv.split(config.n_embd, dim=2) # (b,t, n_embd) each | (b,t, nh * hs)
        k = k.view(B, T, config.n_head, C // config.n_head).transpose(1,2) # (b,t,n_embd) -> (b,t,nh,hs) -> (b,nh,t,hs)
        q = q.view(B, T, config.n_head, C // config.n_head).transpose(1,2) # (b,nh,t,hs)
        v = v.view(B, T, config.n_head, C // config.n_head).transpose(1,2) # (b,nh,t,hs)

        attn = q @ k.transpose(-2, -1) * (1/torch.sqrt(k.size(-1))) # (b,nh,t,hs) @ (b,nh,hs,t) --> (b,nh,t,t)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v # (b,nh,t,t) @ (b,nh,t,hs) -> (b,nh,t,hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x +  self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

model = GPT2(config)
sd = model.state_dict()
for k,v in sd.items():
    print(f'{k} --> {v.shape}')