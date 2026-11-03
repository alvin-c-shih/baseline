# %% [markdown]
# # Module 2 · The Night Line: Sequences
# **Verse 2 + pre-chorus "MEMORY" + chorus 2 "GONE" · 1:00–1:59**
#
# Images have a fixed size. Text, audio and code are **sequences** of varying length, where order matters. Verse 2 is a night train: each car is one sung word, and four generations of machinery ride along it. You'll build each of them and score them all on the same held-out text:
#
# n-gram counting → RNN (and why it forgets) → LSTM gates → seq2seq's bottleneck → attention → the Transformer.
#
# **Runtime:** about 10–15 minutes on the `laptop` tier on CPU, much less on a GPU.

# %%
import sys; sys.path.insert(0, '.')
from blcourse import *
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
hardware_report()

# %% [markdown]
# ## 2.1 Tokens: turning text into integers
# Models consume numbers, so text is first split into **tokens**, each mapped to an integer ID. Our own models will use the simplest tokenizer, one token per character. Production LLMs use **byte-pair encoding (BPE)**, which merges frequent character runs into sub-word tokens. Here's how a real LLM tokenizer sees the chorus line (the first run downloads a few MB):

# %%
from transformers import AutoTokenizer
llm_tok = AutoTokenizer.from_pretrained(CFG['llm'])
line = "You were state of the art, now you're just a baseline."
ids = llm_tok.encode(line)
print(len(line), 'characters →', len(ids), 'BPE tokens')
print(' | '.join(llm_tok.decode([i]) for i in ids))
print(ids)

# %% [markdown]
# Our corpus is *Tiny Shakespeare* (about 1 MB of public-domain plays). We hold out the last 10% as the **validation** set, and every model in this module is scored on it in the same unit: **<abbr title="Minus the log of the probability the model gave to the right answer, averaged.">cross-entropy</abbr> in <abbr title="A unit for measuring surprise, like bits but using the natural log: 1 nat ≈ 1.44 bits.">nats</abbr> per character**. Lower is better.
#
# *What's a nat?* It's a unit of surprise, like a bit but using the natural log (1 nat ≈ 1.44 bits). A loss of L nats means the model is about as unsure as if it were choosing among e<sup>L</sup> equally likely characters. That number, e<sup>L</sup>, is called the **perplexity**. Random guessing among the vocabulary scores ln(vocab size).

# %%
text = shakespeare()
data = CharData(text)
print(f'{len(text):,} chars · vocab {data.vocab} · random-guess loss ln({data.vocab}) = {math.log(data.vocab):.2f} nats/char')
print(text[:250])
SCORES = {}      # model name -> validation loss; we fill this in as we go

# %% [markdown]
# ## 2.2 N-gram counting: a window three tokens wide

# %%
still(62.5)

# %% [markdown]
# The oldest language model is a lookup table. Count how often each character follows each context of n−1 characters, then predict by those frequencies:
#
# $$P(c \mid \text{ctx}) = \frac{\text{count}(\text{ctx}, c) + k}{\text{count}(\text{ctx}) + kV}$$
#
# The `+ k` (**smoothing**) keeps unseen combinations from getting probability zero, which would make the loss infinite.

# %%
train_txt = data.decode(data.train.tolist()); val_txt = data.decode(data.val[:100_000].tolist())

def ngram_loss(n, k=0.05):
    counts = defaultdict(Counter)
    for i in range(n - 1, len(train_txt)):
        counts[train_txt[i - n + 1:i]][train_txt[i]] += 1
    V, nll = data.vocab, 0.0
    for i in range(n - 1, len(val_txt)):
        c = counts.get(val_txt[i - n + 1:i])
        num = (c[val_txt[i]] if c else 0) + k
        den = (sum(c.values()) if c else 0) + k * V
        nll -= math.log(num / den)
    return nll / (len(val_txt) - n + 1), len(counts)

for n in range(1, 7):
    loss, ctxs = ngram_loss(n)
    SCORES[f'{n}-gram'] = loss
    print(f'{n}-gram: val {loss:.3f} nats/char   ({ctxs:,} distinct contexts)')

# %% [markdown]
# A longer window helps up to a point. After that the table becomes too **sparse**: most long contexts in the validation text never appeared in training, so the counts carry no information. Counting can't generalize from "the king" to "the queen". A model that represents tokens as vectors of numbers (**embeddings**) can, because similar tokens end up with similar vectors.

# %% [markdown]
# ## 2.3 RNN: carry a hidden state, and backprop through time

# %%
still(64.6)

# %% [markdown]
# A **recurrent neural network** reads one token at a time and updates a fixed-size vector, the **hidden state**:
#
# $$h_t = \tanh(W_{hh}\,h_{t-1} + W_{xh}\,x_t + b)$$
#
# In principle h carries everything seen so far. To train it, you unroll it over the sequence and backprop through the unrolled graph (**backprop through time**, BPTT). The gradient that reaches step t from step T has passed through W_hh about T − t times, so it gets multiplied by roughly the same matrix over and over. If W_hh's largest stretch factor (its spectral radius ρ) is below 1, the gradient **vanishes**. Above 1, it can **explode**.

# %%
def bptt_profile(rho, T=64, d=32):
    seed(0)
    W = torch.randn(d, d); W *= rho / torch.linalg.eigvals(W).abs().max()
    hs, h = [], torch.zeros(d, requires_grad=True)
    for t in range(T):
        h = W @ h + 0.1 * torch.randn(d)        # linear RNN: the cleanest view of the effect
        h.retain_grad(); hs.append(h)
    hs[-1].sum().backward()
    return [x.grad.norm().item() for x in hs]

fig, ax = plt.subplots()
for rho, c in ((0.8, ASH), (1.0, BONE), (1.2, MAG)):
    ax.plot(range(64, 0, -1), bptt_profile(rho), c=c, label=f'ρ = {rho}')
ax.set(yscale='log', xlabel='steps back from the loss', ylabel='‖∂L/∂h_t‖', title='BACKPROP THROUGH TIME'); ax.legend();

# %%
still(70.5)

# %% [markdown]
# Exploding gradients are easy to fix: **clip** the gradient norm (`clip_grad_norm_`, which every training loop in this course uses). Vanishing gradients are harder. They mean a plain RNN can't learn dependencies more than a few dozen steps long.
#
# ## 2.4 LSTM: put gates along the road (1997)
# The LSTM keeps a separate **cell state** c that runs alongside the sequence like a highway, and three learned **gates** (values between 0 and 1) decide what happens to it at each step:
#
# - **forget** f: how much of the old cell state to keep,
# - **input** i: how much new information to write,
# - **output** o: how much of the cell to reveal as the hidden state.
#
# $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t \qquad h_t = o_t \odot \tanh(c_t)$$
#
# Because c is updated by *multiplying by f and adding*, rather than by being squashed through a matrix every step, the gradient along it can survive for hundreds of steps whenever f ≈ 1.

# %%
class CharLSTM(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
    def forward(self, idx, targets=None):
        h, _ = self.lstm(self.emb(idx))
        logits = self.head(h)
        return logits, None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())

L = CFG['lstm']
seed(0); lstm = CharLSTM(data.vocab, L['hidden'])
print(f'LSTM: {n_params(lstm):,} parameters')
lstm_hist = train_lm(lstm, data, L['steps'], L['block'], L['batch'], lr=4e-3, label='LSTM')
SCORES['LSTM'] = eval_loss(lstm, data, L['block'], iters=50)['val']

# %% [markdown]
# **Watch the two columns.** Train loss keeps falling. On the bigger tiers, validation loss bottoms out and then rises: the model has started memorizing Tiny Shakespeare (only about 1 MB) instead of learning patterns that transfer. That's Module 0's overfitting curve, live. `train_lm` keeps the weights from the checkpoint with the lowest validation loss and restores them at the end. That's **early stopping**, the validation set doing the job Module 0 gave it. The bigger tiers also use **dropout**, which randomly zeroes some activations during training so no single path can memorize.
#
# ### Look at the gates
# `nn.LSTM` doesn't return its gates, but they're just a function of its weights, so we can recompute them for a line of text. Each row below is one gate unit, each column one character.

# %%
@torch.no_grad()
def lstm_gates(model, s):
    W_ih, W_hh = model.lstm.weight_ih_l0.cpu(), model.lstm.weight_hh_l0.cpu()
    b = (model.lstm.bias_ih_l0 + model.lstm.bias_hh_l0).cpu()
    x = model.emb.cpu()(torch.tensor(data.encode(s))); model.to(DEVICE)
    H = W_hh.shape[1]; h = c = torch.zeros(H); out = {k: [] for k in 'fio'}
    for xt in x:
        z = W_ih @ xt + W_hh @ h + b
        i, f, g, o = torch.sigmoid(z[:H]), torch.sigmoid(z[H:2*H]), torch.tanh(z[2*H:3*H]), torch.sigmoid(z[3*H:])
        c = f * c + i * g; h = o * torch.tanh(c)
        for k, v in zip('fio', (f, i, o)): out[k].append(v)
    return {k: torch.stack(v).T for k, v in out.items()}

s = 'ROMEO:\nBut soft, what light through yonder window breaks?'
g = lstm_gates(lstm, s)
fig, axes = plt.subplots(3, 1, figsize=(11, 4.5), sharex=True)
for ax, k, name in zip(axes, 'fio', ('FORGET', 'INPUT', 'OUTPUT')):
    ax.imshow(g[k][:24], aspect='auto', vmin=0, vmax=1, cmap='baseline'); ax.set_ylabel(name, fontsize=8); ax.set_yticks([]); ax.grid(False)
axes[-1].set_xticks(range(len(s))); axes[-1].set_xticklabels([c if c != '\n' else '⏎' for c in s], fontsize=7)
fig.suptitle('LSTM GATES · 24 UNITS × 1 LINE', x=0.01, ha='left', color=BONE, fontweight='bold');

# %%
still(74.0)

# %% [markdown]
# ## 2.5 Seq2seq squeezes a sentence through a bottleneck; attention looks back
# Translation maps one sequence to another of a different length. **Seq2seq** (Sutskever et al., 2014) uses two RNNs. An *encoder* reads the whole input into one fixed-size vector, and a *decoder* generates the output from that vector alone. That vector is the **bottleneck**: a 5-word sentence and a 50-word sentence must fit in the same space.
#
# **Attention** (Bahdanau et al., 2014) removes it. At every output step the decoder compares its state to *every* encoder state, turns the similarities into weights with a softmax, and reads a weighted average of them:
#
# $$\alpha_{tj} = \text{softmax}_j(s_t \cdot h_j) \qquad \text{context}_t = \sum_j \alpha_{tj}\,h_j$$
#
# Test task: **reverse a string of digits**. It's trivial for code, and a good stress test for memory.

# %%
still(77.2)

# %%
V_DIG, BOS, MAXLEN = 12, 10, 24                 # tokens 0-9 digits, 10 = start, 11 = pad

def reverse_batch(B, lo=4, hi=MAXLEN):
    n = torch.randint(lo, hi + 1, (B,))
    src = torch.full((B, hi), 11); tgt = torch.full((B, hi), 11)
    for b in range(B):
        d = torch.randint(0, 10, (n[b],)); src[b, :n[b]] = d; tgt[b, :n[b]] = d.flip(0)
    return src, tgt, n

class Seq2Seq(nn.Module):
    def __init__(self, hidden=128, attention=False):
        super().__init__()
        self.attention = attention
        self.emb = nn.Embedding(V_DIG, 32)
        self.enc = nn.GRU(32, hidden, batch_first=True)
        self.dec = nn.GRU(32, hidden, batch_first=True)
        self.out = nn.Linear(hidden * (2 if attention else 1), V_DIG)
    def forward(self, src, tgt_in):
        H, h = self.enc(self.emb(src))                       # H: all encoder states; h: the final one (the bottleneck)
        S, _ = self.dec(self.emb(tgt_in), h)
        if not self.attention:
            return self.out(S), None
        scores = S @ H.transpose(1, 2)                        # (B, T_out, T_in)
        scores = scores.masked_fill((src == 11).unsqueeze(1), -1e9)
        alpha = scores.softmax(-1)
        return self.out(torch.cat([S, alpha @ H], -1)), alpha

def shift(tgt):
    return torch.cat([torch.full((len(tgt), 1), BOS, device=tgt.device), tgt[:, :-1]], 1)

models = {}
for attention in (False, True):
    seed(0); m = Seq2Seq(attention=attention).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    with timer(f"seq2seq {'+ attention' if attention else '(bottleneck)'}:"):
        for step in range(1500):
            src, tgt, _ = reverse_batch(64)
            logits, _ = m(src.to(DEVICE), shift(tgt).to(DEVICE))
            loss = F.cross_entropy(logits.flatten(0, 1), tgt.to(DEVICE).flatten(), ignore_index=11)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step()
    models[attention] = m

@torch.no_grad()
def decode(m, src, n):                                        # greedy, one token at a time
    tgt = torch.full((len(src), MAXLEN), 11, device=DEVICE)
    for t in range(n):
        logits, alpha = m(src, shift(tgt))
        tgt[:, t] = logits[:, t].argmax(-1)
    return tgt[:, :n], alpha

lengths = list(range(4, MAXLEN + 1, 2)); acc = {False: [], True: []}
for n in lengths:
    src, tgt, _ = reverse_batch(200, n, n)
    src = torch.cat([src[:, :n], torch.full((200, MAXLEN - n), 11)], 1).to(DEVICE)
    for att in (False, True):
        pred, _ = decode(models[att], src, n)
        acc[att].append((pred.cpu() == tgt[:, :n]).all(1).float().mean().item())

fig, ax = plt.subplots()
ax.plot(lengths, acc[False], 'o-', c=ASH, label='seq2seq (bottleneck)')
ax.plot(lengths, acc[True], 'o-', c=MAG, label='seq2seq + attention')
ax.set(xlabel='sequence length', ylabel='exact-match accuracy', title='REVERSE A DIGIT STRING', ylim=(-0.03, 1.03)); ax.legend();

# %% [markdown]
# The bottleneck model degrades as the input gets longer. The attention model doesn't, because it never has to fit the whole sequence into one vector. Here is what it looks at while reversing one string:

# %%
digits = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8] + [11] * (MAXLEN - 12)], device=DEVICE)
pred, alpha = decode(models[True], digits, 12)
fig, ax = plt.subplots(figsize=(4.4, 4.4))
ax.imshow(alpha[0, :12, :12].cpu(), cmap='baseline'); ax.grid(False)
ax.set(xticks=range(12), yticks=range(12), xlabel='input position', ylabel='output step', title='ATTENTION α · REVERSAL')
ax.set_xticklabels(digits[0, :12].tolist()); ax.set_yticklabels(pred[0].tolist());

# %%
still(81.8)

# %% [markdown]
# ## 2.6 The Transformer: attention is all you need (2017)
# If attention does the important work, do we need the recurrence at all? The **Transformer** (Vaswani et al., 2017) drops it. Each layer is **self-attention**, where every position attends to every other position in the same sequence, followed by a small per-position MLP. There's no sequential dependency between positions, so a GPU computes all of them **in parallel**. That parallelism is why it scaled when RNNs couldn't.
#
# $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
#
# Q, K and V are three learned linear projections of the input. A **decoder-only** language model like GPT adds a *causal mask*, so position t can only attend to positions ≤ t: it predicts the next token from the past. `TinyGPT` in `blcourse.py` is about 50 lines; read it.

# %%
still(88.0)

# %%
G = CFG['gpt']
seed(0); gpt = TinyGPT(data.vocab, **G)
print(f'TinyGPT: {n_params(gpt):,} parameters')
gpt_hist = train_lm(gpt, data, G['steps'], G['block'], G['batch'], label='GPT')
SCORES['Transformer'] = eval_loss(gpt, data, G['block'], iters=50)['val']

# %%
ctx = torch.tensor([data.encode('ROMEO:\n')], device=DEVICE)
print(data.decode(gpt.generate(ctx, 300)[0].tolist()))

# %% [markdown]
# ### The scoreboard
# All models, the same validation text, the same unit. This is the module's own leaderboard.

# %%
names = list(SCORES); vals = [SCORES[k] for k in names]
best = min(vals)
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.barh(names, vals, color=[MAG if v == best else GRAPHITE for v in vals])
for i, v in enumerate(vals):
    ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=8, color=BONE)
ax.invert_yaxis(); ax.set(xlabel='validation loss (nats/char) · lower is better', title='THE BOARD · TINY SHAKESPEARE')
ax.axvline(math.log(data.vocab), c=ASH, ls=':', lw=1);

# %% [markdown]
# On the `laptop` tier the LSTM and the Transformer often finish close together. At this size and training budget the Transformer has no big edge in modeling ability. Its edge is that it **parallelizes and scales**, which you'll measure in Module 5. Try the `mac-max`/`cuda` tiers and watch the gap open.
#
# ### What does a head look at?
# Attention weights for the last layer's heads on one line of text. Each row is a position doing the looking. The upper triangle is empty because of the causal mask.

# %%
@torch.no_grad()
def attn_maps(model, s, layer=0):
    model.eval()
    x = model.embed(torch.tensor([data.encode(s)], device=DEVICE))
    for b in model.blocks[:layer]:
        x = b(x)
    return model.blocks[layer].attn_weights(x)[0].cpu()

s = 'To be, or not to be, that is the question:'
w = attn_maps(gpt, s, layer=len(gpt.blocks) - 1)
fig, axes = plt.subplots(1, w.shape[0], figsize=(3 * w.shape[0], 3.2))
for h_, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(w[h_] ** 0.5, cmap='baseline'); ax.set_title(f'HEAD {h_}'); ax.axis('off')

# %%
still(113.5)

# %% [markdown]
# ### Checkpoint
# 1. Why does an n-gram model stop improving as n grows?
# 2. What goes wrong when you backprop through 64 steps of an RNN, and how do LSTM gates help?
# 3. What is the seq2seq bottleneck? Sketch how attention removes it.
# 4. Why is the Transformer easier to scale than an LSTM, even when they reach similar loss at small sizes?
#
# **Next:** Module 3, the parts of a modern LLM.
