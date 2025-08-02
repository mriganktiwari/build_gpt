import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size = 1024
    vocab_size = 50257
    n_layer = 12
    n_head = 12
    n_embd = 768
config = GPTConfig

class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # q,k,v all in a big tensor
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # this is tril in our gpt implementation, not really 'bias'
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size,config.block_size)
                            )

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x) # (b, t ,3*n_embd)
        # at every time step: vector for q,k,v are concatenated here
        q,k,v = qkv.split(self.n_embd, dim=2) # (b,t,n_embd) each for q,k,v
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (b,t,n_head,head_dim) -> (b,n_head,t,head_dim)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (b,t,n_head,head_dim) -> (b,n_head,t,head_dim)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (b,t,n_head,head_dim) -> (b,n_head,t,head_dim)

        att = q @ k.transpose(-1,-2) * (1. / math.sqrt(k.size(-1))) # (b,n_head,t,head_dim) @ (b,n_head,head_dim,t) --> (b,n_head,t,t)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v # (b,n_head,t,t) @ (b,n_head,t,head_dim) -> (b,n_head,t,head_dim)
        y = y.transpose(1,2).contiguous().view(B,T,C) # (b,t,n_embd)
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
        self.attn = CausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
