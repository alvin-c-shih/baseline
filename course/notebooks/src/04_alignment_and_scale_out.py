# %% [markdown]
# # Module 4 · Both Ways, the Plant, the Tape
# **Chorus 3b + bridge · 2:43–3:19**
#
# By 2020 the architecture had settled: a decoder-only Transformer, scaled up. The bridge covers what happened next: making those models *helpful* (RLHF, then DPO), making attention *fast* (FlashAttention), and making context *long* (2K → 1M tokens, checked with needle-in-a-haystack tests).
#
# You'll reuse the LLM from Module 3. It's cached, so it loads quickly this time.

# %%
import sys; sys.path.insert(0, '.')
from blcourse import *
import matplotlib.pyplot as plt
hardware_report()

# %% [markdown]
# ## 4.1 BERT read both ways; decoder-only scaled up
# **BERT** (2018) is a Transformer *encoder*. Every token attends to every other token, left and right, and it's trained by hiding 15% of the tokens behind `[MASK]` and predicting them. That made it excellent at *understanding* tasks (classification, search, extraction), but it can't generate text left to right.
#
# A **decoder-only** model (GPT) uses a causal mask and predicts the next token. Scaled to 175B parameters (GPT-3, 2020), it handled most of BERT's tasks through prompting, and it could generate as well. The only architectural difference is the mask:

# %%
n = 8
fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
for ax, (title, m) in zip(axes, [('BERT · BIDIRECTIONAL', np.ones((n, n))), ('GPT · CAUSAL', np.tril(np.ones((n, n))))]):
    ax.imshow(m, cmap='baseline', vmin=0, vmax=1.4); ax.set_title(title); ax.set(xlabel='attends to', ylabel='token'); ax.grid(False)

# %%
still(164.0)

# %% [markdown]
# Let's ask a real BERT (DistilBERT, a 260 MB download) to fill in the film's mask:

# %%
from transformers import pipeline
fill = pipeline('fill-mask', model='distilbert/distilbert-base-uncased', device=DEVICE)
for r in fill('you were [MASK] of the art, now you are just a baseline.', top_k=5):
    print(f"{r['score']:.3f}  {r['token_str']}")
del fill

# %%
still(168.2)

# %% [markdown]
# BERT-style encoders are far from dead in practice. They're still the workhorse behind **embedding models** for search and retrieval. But "SOTA on GLUE" went to the decoders.
#
# ## 4.2 RLHF trained a reward model to rank
# A pretrained LLM continues text. That's not the same as being helpful. **RLHF** (Christiano 2017; InstructGPT 2022) adds three steps:
#
# 1. **SFT**: fine-tune on human-written example answers (what you did with LoRA in Module 3).
# 2. **Reward model**: humans compare pairs of answers ("A is better than B"), and a model learns to score answers so that the scores predict those preferences.
# 3. **RL (PPO)**: optimize the LLM to maximize the reward model's score, with a penalty for drifting too far from the SFT model.
#
# Step 2 rests on the **Bradley–Terry** model of comparisons: P(A ≻ B) = σ(r_A − r_B). Below, four answers have hidden "true" quality. We simulate 300 noisy human votes and recover the scores from the votes alone.

# %%
still(172.5)

# %%
seed(0)
true_r = torch.tensor([1.1, 0.2, 1.9, -0.8])                      # answers A, B, C, D
pairs = torch.randint(0, 4, (600, 2)); pairs = pairs[pairs[:, 0] != pairs[:, 1]][:300]
p_first = torch.sigmoid(true_r[pairs[:, 0]] - true_r[pairs[:, 1]])
first_wins = torch.rand(len(pairs)) < p_first                     # noisy human labels

r = torch.zeros(4, requires_grad=True); opt = torch.optim.Adam([r], 0.05)
for _ in range(500):
    margin = r[pairs[:, 0]] - r[pairs[:, 1]]
    loss = F.binary_cross_entropy_with_logits(margin, first_wins.float())   # Bradley–Terry likelihood
    opt.zero_grad(); loss.backward(); opt.step()
fitted = (r - r.mean()).detach(); truth = true_r - true_r.mean()
for name, t, f_ in zip('ABCD', truth, fitted):
    print(f'answer {name}: true {t:+.2f}   fitted from votes {f_:+.2f}')
print('ranking:', ' ≻ '.join('ABCD'[i] for i in fitted.argsort(descending=True)))

# %% [markdown]
# ## 4.3 DPO skipped the middleman
# RLHF's step 3 is the fiddly one: an RL loop, a separate reward model held in memory, and a lot of tuning. **DPO** (Rafailov et al., 2023) shows that, for the KL-constrained objective RLHF optimizes, the best policy has a closed form. Substituting it back into the Bradley–Terry likelihood cancels the intractable normalizer Z(x), and leaves a plain supervised loss on preference pairs (y_w preferred over y_l):
#
# $$L_\text{DPO} = -\log\sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\big]\Big)$$
#
# No reward model and no RL loop. You need two forward passes per pair, through the model being trained (π_θ) and through a frozen copy (π_ref). With LoRA the frozen copy is free: it's the same model with the patch turned off (`lora_disabled`).
#
# Our preference: **short, direct answers.** For each question, the *chosen* answer is one sentence we wrote. The *rejected* answer is whatever the model itself says now. That's how real preference data works: you compare against the model's own outputs, so it learns to move away from what it actually does.

# %%
still(176.5)

# %%
PREFS = [                                  # (question, the short answer we prefer)
    ('What is overfitting?', 'When a model memorizes its training data instead of learning patterns that generalize.'),
    ('What is a learning rate?', 'The step size gradient descent takes when updating the weights.'),
    ('What does a GPU do for deep learning?', 'It runs the many matrix multiplications of training and inference in parallel.'),
    ('What is a token?', 'A chunk of text, often part of a word, that the model reads as one integer.'),
    ('What is a loss function?', 'A number that measures how wrong the model is; training minimizes it.'),
    ('What is an embedding?', 'A learned vector that represents a token or item so that similar things are close.'),
    ('What is a validation set?', 'Held-out data used to choose settings, kept separate from the final test set.'),
    ('What is fine-tuning?', 'Continuing to train a pretrained model on your own, smaller dataset.'),
    ('What is attention?', 'A weighted average over all positions, with weights from query–key similarity.'),
    ('What is quantization?', 'Storing weights in fewer bits, like 4 instead of 16, to save memory and time.'),
    ('What is a GAN?', 'Two networks, a generator and a discriminator, trained against each other.'),
    ('What is a benchmark?', 'A fixed public test set and metric that models are compared on.'),
]
HELD = ['What is backpropagation?', 'What is a context window?', 'What is a mixture of experts?', 'What is dropout?']

tok, llm = load_llm()
def answers(tag):
    out = {q: chat(tok, llm, q, 120) for q in HELD}
    print(f'{tag}: mean {np.mean([len(a.split()) for a in out.values()]):.0f} words')
    for q, a in out.items():
        print(f'  {q}\n    {a[:200]}')
    return out
before = answers('BEFORE DPO')
own = {q: chat(tok, llm, q, 120) for q, _ in PREFS}         # the rejected side: the model's current answers

# %%
lora_layers = add_lora(llm, r=8)
opt = torch.optim.AdamW([p for p in llm.parameters() if p.requires_grad], lr=2e-4)
beta, log = 0.1, []
seed(0)
with timer('DPO:'):
    for step in range(CFG['dpo_steps']):
        q, good = PREFS[step % len(PREFS)]
        ids_w, n = chat_ids(tok, q, good); ids_l, _ = chat_ids(tok, q, own[q])
        with torch.no_grad(), lora_disabled(lora_layers):                    # π_ref: the model without the patch
            ref_w, ref_l = answer_logprob(llm, ids_w, n), answer_logprob(llm, ids_l, n)
        pol_w, pol_l = answer_logprob(llm, ids_w, n), answer_logprob(llm, ids_l, n)
        margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))                   # implicit reward difference
        loss = -F.logsigmoid(margin).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        log.append((loss.item(), margin.item()))
fig, ax = plt.subplots(figsize=(6, 2.6))
ax.plot([m for _, m in log], c=MAG); ax.axhline(0, c=GRAPHITE, lw=1)
ax.set(title='DPO IMPLICIT REWARD MARGIN (CHOSEN − REJECTED)', xlabel='step');

# %%
after = answers('AFTER DPO')

# %%
still(174.8)

# %% [markdown]
# A dozen preference pairs and a few dozen steps on a laptop visibly shorten the model's answers to questions it never saw. Shorter isn't the same as correct, though: a 0.5B model's facts are still shaky.
#
# ### Try it
# Rerun the DPO cells with `lr=5e-4`. The margin climbs higher, and the answers get shorter still, then stop making sense. The model has **over-optimized** the preference and drifted far from what it knew. That's why RLHF penalizes drift from the reference model, and why β and the step budget matter. A preference signal tells the model which direction to move, not how far.
#
# This is how the "personality" of chat models is shaped, with far more data and far more care. Notice also how *easy* it would be to teach a bad preference. Alignment data is a design decision, not a detail.
#
# ## 4.4 FlashAttention tiles it all inside the SRAM
# Standard attention computes the full n × n score matrix S = QKᵀ, writes it to GPU main memory (HBM), reads it back for the softmax, writes the result, and reads it again to multiply by V. For n = 32K that's 1 billion entries per head. The arithmetic is cheap. The **memory traffic** is the bottleneck.
#
# FlashAttention is built from two ideas that have nothing to do with ML. Try each one on its own first.
#
# ### Stepping stone 1: a matrix multiply works in tiles
# Each cell of C = A·B is a row of A dotted with a column of B. Split the shared dimension into blocks and add up the block products, and you get the same C, up to float rounding, because the additions happen in a different order. But each block is small enough to sit in fast memory while you use it. CPU math libraries call this **loop tiling** (or cache blocking) and have done it for decades.

# %%
seed(0)
A, B = torch.randn(256, 512), torch.randn(512, 128)
C_tiled = torch.zeros(256, 128)
for k0 in range(0, 512, 64):                      # walk the shared dimension 64 columns at a time
    C_tiled += A[:, k0:k0 + 64] @ B[k0:k0 + 64, :]  # each partial product only needs one tile of A and one of B
print('max |A@B − tiled| =', (A @ B - C_tiled).abs().max().item())

# %% [markdown]
# ### Stepping stone 2: a softmax as running totals
# A softmax needs two things about its row: the max (subtracted first so `exp` can't overflow) and the sum of exps. You can get both from a stream, one chunk at a time, without storing the stream. Just keep a running max and a running sum. When a new chunk raises the max, the sum you already have was computed against the old max, so multiply it by e^(old max − new max) to bring it up to date.

# %%
scores = torch.randn(10_000) * 3
m, l = -float('inf'), 0.0
for chunk in scores.split(1000):                   # pretend the scores arrive 1,000 at a time
    m_new = max(m, chunk.max().item())
    l = l * math.exp(m - m_new) + torch.exp(chunk - m_new).sum().item()   # rescale the old sum, add the new chunk
    m = m_new
print(f'streamed: max {m:.4f}  sum {l:.4f}')
print(f'all at once: max {scores.max().item():.4f}  sum {torch.exp(scores - scores.max()).sum().item():.4f}')

# %% [markdown]
# ### Putting them together
# **FlashAttention** (Dao et al., 2022) never stores S. It walks over K and V one tile at a time, like stepping stone 1, inside the GPU's small on-chip SRAM. For each query row it keeps the running max m and running sum ℓ from stepping stone 2 (the **online softmax**), plus a running weighted sum of V that gets the same rescaling. The result is *exactly* the same attention, with memory linear in n. Here it is in plain PyTorch:

# %%
still(179.0)

# %%
def naive_attention(q, k, v):
    S = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])          # n × n, materialized
    return S.softmax(-1) @ v

def tiled_attention(q, k, v, tile=128):
    """Online-softmax attention over K/V tiles. Never holds more than n × tile scores."""
    scale = 1 / math.sqrt(q.shape[-1])
    m = torch.full(q.shape[:-1], -float('inf'), device=q.device)      # running row max
    l = torch.zeros(q.shape[:-1], device=q.device)                   # running row sum of exp
    acc = torch.zeros_like(q)
    for j in range(0, k.shape[-2], tile):
        s = (q @ k[..., j:j + tile, :].transpose(-1, -2)) * scale     # n × tile
        m_new = torch.maximum(m, s.amax(-1))
        p = torch.exp(s - m_new[..., None])
        corr = torch.exp(m - m_new)                                   # rescale what we had so far
        l = l * corr + p.sum(-1)
        acc = acc * corr[..., None] + p @ v[..., j:j + tile, :]
        m = m_new
    return acc / l[..., None]

seed(0)
q, k, v = (torch.randn(1, 4, 1024, 64, device=DEVICE) for _ in range(3))
print('max |naive − tiled| =', (naive_attention(q, k, v) - tiled_attention(q, k, v)).abs().max().item())
print('max |naive − torch SDPA| =', (naive_attention(q, k, v) - F.scaled_dot_product_attention(q, k, v)).abs().max().item())

# %% [markdown]
# Same numbers. Now the costs. `F.scaled_dot_product_attention` (SDPA) is PyTorch's fused kernel, which uses FlashAttention-style kernels on CUDA and an efficient kernel on MPS/CPU. The pure-Python tiled version shows the *algorithm*. It isn't fast, because Python can't keep the tile in SRAM. That's why FlashAttention is written as a custom GPU kernel.

# %%
def bench(fn, n, H=4, d=64, reps=3):
    q, k, v = (torch.randn(1, H, n, d, device=DEVICE) for _ in range(3))
    try:
        fn(q, k, v); sync(); t0 = time.perf_counter()
        for _ in range(reps):
            fn(q, k, v)
        sync(); return (time.perf_counter() - t0) / reps * 1e3
    except (RuntimeError, MemoryError):                               # out of memory is itself the lesson
        return float('nan')

rows = []
for n in CFG['attn_lengths']:
    rows.append((n, bench(naive_attention, n), bench(F.scaled_dot_product_attention, n), 4 * n * n * 4 / 2**20))
    print(f'n={n:6d}  naive {rows[-1][1]:8.1f} ms   SDPA {rows[-1][2]:8.1f} ms   naive score matrix {rows[-1][3]:8.0f} MB')
ns = [r[0] for r in rows]
fig, ax = plt.subplots()
ax.plot(ns, [r[1] for r in rows], 'o-', c=ASH, label='naive (materializes n×n)'); ax.plot(ns, [r[2] for r in rows], 'o-', c=MAG, label='fused SDPA')
ax.set(xscale='log', yscale='log', xlabel='sequence length n', ylabel='ms per call', title=f'ATTENTION · 4 HEADS · {DEVICE.upper()}'); ax.legend();

# %%
still(183.0)

# %% [markdown]
# ## 4.5 Context window: 2K, then a million tokens long
# GPT-3 read 2,048 tokens at a time. Gemini 1.5 (2024) read a million. Longer context needed cheaper attention (above), better position encodings (RoPE with scaled frequencies), and long-document training data. But does a model actually *use* a long context?
#
# **Needle in a haystack**: hide one fact at some depth inside long filler text, then ask for it. Sweep context length × depth. Your tier's LLM and lengths are below. Wherever your model starts missing needles, that is the result.

# %%
still(189.2)

# %%
NEEDLE = 'The secret password for the vault is MAGENTA-1958.'
QUESTION = 'What is the secret password for the vault? Reply with the password only.'
hay_ids = tok(shakespeare()[:400_000])['input_ids']

def niah(length, depth):
    body = hay_ids[:length]; cut = int(depth * len(body))
    text = tok.decode(body[:cut]) + '\n' + NEEDLE + '\n' + tok.decode(body[cut:])
    out = chat(tok, llm, f'{text}\n\n{QUESTION}', 12)
    return 'MAGENTA-1958' in out.upper().replace(' ', ''), out

with lora_disabled(lora_layers):
    grid = np.zeros((len(CFG['niah_depths']), len(CFG['niah_lengths'])))
    with timer('needle in a haystack:'):
        for j, L in enumerate(CFG['niah_lengths']):
            for i, d in enumerate(CFG['niah_depths']):
                ok, out = niah(L, d); grid[i, j] = ok
            print(f'{L:6d} tokens: {int(grid[:, j].sum())}/{len(CFG["niah_depths"])} found   (last answer: {out!r})')

from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.imshow(grid, cmap=ListedColormap([MAG_DARK, MAG]), vmin=0, vmax=1, aspect='auto'); ax.grid(False)
ax.set(xticks=range(len(CFG['niah_lengths'])), xticklabels=CFG['niah_lengths'], yticks=range(len(CFG['niah_depths'])),
       yticklabels=[f'{d:.0%}' for d in CFG['niah_depths']], xlabel='context length (tokens)', ylabel='needle depth',
       title=f'NEEDLE IN A HAYSTACK · {CFG["llm"].split("/")[-1]}');

# %%
still(191.0)

# %% [markdown]
# Passing needle-in-a-haystack is necessary but nowhere near sufficient. Finding one planted sentence is much easier than *reasoning* over a whole long document. Newer benchmarks (RULER, multi-needle, long-document QA) test that, and they're much harder.
#
# ### Checkpoint
# 1. Encoder vs decoder-only: what's the one architectural difference, and what does each make easy?
# 2. Write the Bradley–Terry probability. What does a reward model learn from it?
# 3. Why doesn't DPO need a reward model? What plays the role of π_ref in our LoRA setup?
# 4. FlashAttention computes the same result as standard attention. So what does it save?
#
# **Next:** Module 5, scaling laws and what comes after quadratic attention.
