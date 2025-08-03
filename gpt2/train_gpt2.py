import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        # x shape: (b,t)
        B,T = x.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long)
        pos_emb = self.transformer.wpe(pos) # (b,t,n_embd)
        tok_emb = self.transformer.wte(x)   # (b,t,n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (b,t,vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        # loads HF pretrained model weights for gpt2
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f'loading weights from pretrained gpt: {model_type}')

        # n_layer, n_head, n_embd are determined form model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        # create a from scratch initialized gpt2 model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the masks/buffers

        # init a HF model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while assuring all parameters match in name and shape
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the masks/buffers
        sd_keys_hf = [k for k in sd_keys if not k.endswith('.attn.masked_bias')] # discard the masks/buffers
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape # ensuring transposed shape of sd_hf[k] == shape of sd[k]
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

# --------------------------------------------------------------------

max_return_sequences = 5
max_length = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)

# create prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(max_return_sequences, 1) # (5,8)
x = tokens.to(device=device)

# generate
torch.manual_seed(42)
while x.size(1) <= max_length:
    with torch.no_grad():
        logits = model(x) # (b,t,vocab_size)
        logits = logits[:, -1, :] # (b,vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1) # (b, 1)
        # look up the real vocabulary token from topk_indices
        xcol = torch.gather(topk_indices, -1, ix) # (b,1)
        x = torch.cat((x,xcol), dim=1) # (b,t+1)

# print the generated text
for i in range(max_return_sequences):
    tokens = x[i, :max_length].tolist() # (max_length,)
    decoded = enc.decode(tokens)
    print("> ", decoded)