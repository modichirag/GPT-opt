# Basically taken from Karpathy's llm.c repo
# https://github.com/karpathy/llm.c/blob/master/train_gpt2.py
import torch
from torch import  nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.split_qkv = config.split_qkv
        if self.split_qkv:
            # separate query, key, value projections, with no query/key bias (needed by ANGD)
            self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.c_v = nn.Linear(config.n_embd, config.n_embd)
        else:
            # key, query, value projections for all heads, but in a batch
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash_attention = config.flash_attention
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # the ANGD optimizer sets angd_capture, and reads the stored input to compute its preconditioner
        self.angd_capture = False
        self.angd_input = None

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.split_qkv:
            if self.angd_capture and self.training:
                self.angd_input = x.detach()
            q, k, v = self.c_q(x), self.c_k(x), self.c_v(x)
        else:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

    @torch.no_grad()
    def angd_fisher_diag(self, x, chunk_size=4):
        """
        Diagonal of the Fisher information of the attention distributions w.r.t.
        c_q.weight and c_k.weight, averaged over all queries in the batch. Each head
        uses the single-head formulas on its own rows of c_q.weight and c_k.weight:
            D_K = (Q*Q) V_X^T / hs,  V_X = E_p[x*x] - E_p[x]*E_p[x]
            D_Q = V_K (X*X)^T / hs,  V_K = E_p[k*k] - E_p[k]*E_p[k]
        where E_p averages keys under each query's attention distribution.
        Args:
            x: Input to the attention layer, shape (B, T, C).
            chunk_size: Number of sequences processed at once, to bound memory.
        Returns:
            D_Q, D_K, each with the shape of c_q.weight.
        """
        assert self.split_qkv
        B, T, C = x.size()
        H, hs = self.n_head, C // self.n_head
        dtype = torch.promote_types(x.dtype, torch.float32)
        Wq, Wk = self.c_q.weight.to(dtype), self.c_k.weight.to(dtype)
        mask = self.causal_mask[0, 0, :T, :T] == 0
        D_Q = torch.zeros(H, hs, C, dtype=dtype, device=x.device)
        D_K = torch.zeros(H, hs, C, dtype=dtype, device=x.device)

        # Variances are computed as differences E[a^2] - E[a]^2, so avoid TF32 matmuls, and
        # subtract per-sequence means (variance is invariant to shifts) to reduce cancellation.
        matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            for xb in x.to(dtype).split(chunk_size):
                b = xb.size(0)
                q = (xb @ Wq.t()).view(b, T, H, hs).transpose(1, 2) # (b, nh, T, hs)
                k = (xb @ Wk.t()).view(b, T, H, hs).transpose(1, 2) # (b, nh, T, hs)
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
                att = att.masked_fill(mask, float('-inf'))
                P = F.softmax(att, dim=-1) # (b, nh, T, T), rows are queries

                # D_K. The E_p[x*x] term is contracted with Q*Q before multiplying with x*x, which
                # avoids a (b, nh, T, C) intermediate. The E_p[x]^2 term can't be, so we pay for P @ x.
                xc = xb - xb.mean(dim=1, keepdim=True)
                q2 = q.square()
                D_K += torch.einsum('bhai,baj->hij', P.transpose(-2, -1) @ q2, xc.square())
                D_K -= torch.einsum('bhli,bhlj->hij', q2, (P @ xc.unsqueeze(1)).square())

                # D_Q.
                kc = k - k.mean(dim=2, keepdim=True)
                VK = P @ kc.square() - (P @ kc).square()
                D_Q += torch.einsum('bhli,blj->hij', VK, xb.square())
        finally:
            torch.set_float32_matmul_precision(matmul_precision)

        scale = 1.0 / (hs * B * T)
        D_Q = (D_Q * scale).clamp_(min=0).view(C, C)
        D_K = (D_K * scale).clamp_(min=0).view(C, C)
        return D_Q, D_K

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class AttentionBlock(nn.Module):

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

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    split_qkv: bool = False

class GPT(nn.Module):

    def __init__(self, config, device, flash_attention=True):
        super().__init__()
        config.flash_attention = flash_attention
        self.config = config
        self.device = device
       
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([AttentionBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, labels=None, return_logits=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if labels is not None:
            # if we are given some desired labels also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
