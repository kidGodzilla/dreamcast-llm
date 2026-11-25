"""llama2.c-style transformer (RMSNorm, RoPE, SwiGLU, tied embeddings, no biases).
Kept architecturally identical to llama2.c so the Dreamcast C inference stays simple."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    vocab_size: int = 4096
    hidden_dim: int = 768
    max_seq_len: int = 256
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.outer(torch.arange(seq_len).float(), freqs)
    return t.cos(), t.sin()


def apply_rope(x, cos, sin):  # x: (B, T, H, D)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_heads, c.n_kv_heads, c.dim // c.n_heads
        self.wq = nn.Linear(c.dim, c.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(c.dim, c.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(c.dim, c.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(c.n_heads * self.hd, c.dim, bias=False)
        self.w1 = nn.Linear(c.dim, c.hidden_dim, bias=False)
        self.w2 = nn.Linear(c.hidden_dim, c.dim, bias=False)
        self.w3 = nn.Linear(c.dim, c.hidden_dim, bias=False)
        self.attention_norm, self.ffn_norm = RMSNorm(c.dim), RMSNorm(c.dim)
        self.drop = c.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        h = self.attention_norm(x)
        q = apply_rope(self.wq(h).view(B, T, self.nh, self.hd), cos, sin)
        k = apply_rope(self.wk(h).view(B, T, self.nkv, self.hd), cos, sin)
        v = self.wv(h).view(B, T, self.nkv, self.hd)
        if self.nkv != self.nh:
            r = self.nh // self.nkv
            k, v = k.repeat_interleave(r, 2), v.repeat_interleave(r, 2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + self.wo(a.transpose(1, 2).reshape(B, T, -1))
        h = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Transformer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok_embeddings = nn.Embedding(c.vocab_size, c.dim)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.dim)
        self.output = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight  # tied
        cos, sin = rope_tables(c.dim // c.n_heads, c.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, self.cos[:T], self.sin[:T])
        return self.output(self.norm(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_p=0.9, stop=None):
        for _ in range(max_new):
            logits = self(idx[:, -self.c.max_seq_len:])[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, -1)
                sp, si = probs.sort(-1, descending=True)
                cut = sp.cumsum(-1) - sp > top_p
                sp[cut] = 0
                nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
            idx = torch.cat([idx, nxt], 1)
            if stop is not None and nxt.item() in stop:
                break
        return idx
