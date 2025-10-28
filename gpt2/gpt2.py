import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size = 1024
    vocab_size = 50257
    n_embd = 768
    n_head = 12
    n_layer = 12

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.register_buffer('bias',
                             torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size,config.block_size),)
                            #  persistent=False)

    def forward(self, x):
        B,T,C = x.shape
        # 
        qkv = self.c_attn(x) # (b,t,3*n_embd)
        q,k,v = qkv.split(self.n_embd, dim=-1) # (b,t,n_embd) each
        q = q.view(B,T,self.n_head,self.head_size).transpose(1,2) # (b,nh,t,hs)
        k = k.view(B,T,self.n_head,self.head_size).transpose(1,2) # (b,nh,t,hs)
        v = v.view(B,T,self.n_head,self.head_size).transpose(1,2) # (b,nh,t,hs)

        wei = q @ k.transpose(-2, -1) * (1/math.sqrt(k.size(-1))) # (b,nh,t,hs) @ (b,nh,hs,t) -> (b,nh,t,t)
        wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (b,nh,t,t)
        wei = F.softmax(wei, dim=-1) # (b,nh,t,t)

        out = wei @ v # (b,nh,t,t) @ (b,nh,t,hs) -> (b,nh,t,hs)
        return out
 
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
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.sa_heads(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

config = GPTConfig()
model = GPT2(config)
for k,v in model.state_dict().items():
    print(f'{k} --> {v.shape}')
print('\nNamed buffers')
for name, buf in model.named_buffers():
    print(f'{name} --> {buf.shape}')