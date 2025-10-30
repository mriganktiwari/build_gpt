import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.bias: torch.Tensor # added this to hide the type hint squiggly line on `self.bias[:,:,:T,:T]`
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
        x = x + self.attn(self.ln_1(x))
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
    
    @classmethod
    def from_pretrained(cls, model_type):

        # assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'] # O(n) time lookup
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} # O(1) time lookup
        print(f"Loading weights from Huggingface's {model_type}")

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['block_size'] = 1024
        config_args['vocab_size'] = 50257

        config = GPTConfig(**config_args)
        # print(f'n_layer = {config.n_layer}')
        # print(f'n_embd = {config.n_embd}')
        # print(f'n_head = {config.n_head}')

        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        model = GPT2(config=config)
        
        sd_hf = model_hf.state_dict()
        sd = model.state_dict()
        
        sd_keys_hf = list(sd_hf.keys())
        sd_keys = list(sd.keys())
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask used in self-attention
        
        assert sd_keys_hf == sd_keys
        # print(f'keys in sd_hf: {type(list(sd_hf_keys))}')
        # print(f'keys in sd: {type(list(sd_keys))}')
        # for key in list(sd_keys):
        #     if key not in list(sd_hf_keys):
        #         print(key)

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys) == len(sd_keys_hf), f'mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}'

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model


model = GPT2.from_pretrained('gpt2')
print(f'loaded weights !')
# for k,v in model.state_dict().items():
#     print(f'{k} --> {v.shape}')
# print('\nNamed buffers')
# for name, buf in model.named_buffers():
#     print(f'{name} --> {buf.shape}')

# GPT2.from_pretrained('gpt2')