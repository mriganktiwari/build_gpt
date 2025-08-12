import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_embd: int = 384
    n_layer: int = 12
    n_head: int = 12
    block_size: int = 1024

c1 = GPTConfig(n_embd=128)
c2 = GPTConfig()
print(c1 == c2)