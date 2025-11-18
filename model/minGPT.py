import os, math

import torch
import torch.nn as nn
from torch.nn import functional as F


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_pdrop = config.attn_pdrop
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and attn_mask is None:
        # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_pdrop if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, -1e9)
            else:
                # default causal mask
                causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.attn_pdrop, training=self.training)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """A Transformer block, but with a SwiGLU FFN"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # SwiGLU FFN inspired by LLaMA
        # hidden_dim is typically 2/3 of the FFN's intermediate dim
        # FFN intermediate dim is usually 4 * n_embd
        hidden_dim = int(config.n_embd * 4 * 2 / 3)

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # Gated Linear Unit's gate
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.act = nn.SiLU()  # Sigmoid-weighted Linear Unit
        self.dropout = nn.Dropout(config.resid_pdrop)

    def mlpf(self, x):
        # FFN forward pass
        return self.dropout(self.w2(self.act(self.w1(x)) * self.w3(x)))

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer=12,
        n_head=8,
        n_embd=256,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        n_unmasked=0,
    ):
        super().__init__()
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wpe1=nn.Embedding(config.block_size, config.n_embd),
                wpe2=nn.Embedding(config.block_size, config.n_embd),
                wce1=nn.Embedding(4, config.n_embd),
                wce2=nn.Embedding(4, config.n_embd),
                drop1=nn.Dropout(config.embd_pdrop),
                drop2=nn.Dropout(config.embd_pdrop),
                h1=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # decouple with SwiGLU
                h2=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # synth with SwiGLU
                ln_f1=nn.LayerNorm(config.n_embd),
                ln_f2=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self.config = config

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, embeddings, cat_len=None, mode="decouple", attn_mask=None):
        device = embeddings.device
        t = embeddings.shape[1]
        assert (
            t == self.config.block_size
        ), f"Cannot forward, model block size is unmatch. {t} != {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        cat = torch.tensor(cat_len[0] * [0] + cat_len[1] * [1] + cat_len[2] * [2] + cat_len[3] * [3], device=device)

        if mode == "decouple":
            position_embeddings = self.transformer.wpe1(pos)
            catagory_embeddings = self.transformer.wce1(cat)
            x = self.transformer.drop1(embeddings + position_embeddings + catagory_embeddings.unsqueeze(0))
            for block in self.transformer.h1:
                x = block(x) # Decouple doesn't need custom mask
            x = self.transformer.ln_f1(x)
        else:  # synth
            position_embeddings = self.transformer.wpe2(pos)
            catagory_embeddings = self.transformer.wce2(cat)
            x = self.transformer.drop2(embeddings + position_embeddings + catagory_embeddings.unsqueeze(0))
            for block in self.transformer.h2:
                x = block(x, attn_mask=attn_mask)
            x = self.transformer.ln_f2(x)

        logits = self.lm_head(x)
        return x, logits

    def get_block_size(self):
        return self.config.block_size

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
