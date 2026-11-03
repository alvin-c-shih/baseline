# %% [markdown]
# # Module 3 · The Lab: Generation and Efficiency
# **Verse 3 · 1:59–2:28**
#
# Verse 3 moves fast. It covers two ways to *generate* data, the parts list of a modern LLM, and three tricks that make LLMs cheap enough to run and adapt:
#
# GAN → diffusion · sinusoids → RoPE · LayerNorm → RMSNorm · ReLU → SwiGLU · dense → mixture of experts · 32 bits → 4 · full fine-tune → LoRA · then just prompt it in context.
#
# This is the first module that downloads a real pretrained LLM (`CFG['llm']`: Qwen2.5 0.5B on `laptop`, 1.5B on `mac-max`, 7B on `cuda`). Expect 1–15 GB on first run, cached in `~/.cache/huggingface`.

# %%
import sys; sys.path.insert(0, '.')
from blcourse import *
import matplotlib.pyplot as plt
hardware_report()

# %% [markdown]
# ## 3.1 GANs played a two-player game
# A **generative** model learns to produce new samples that look like the training data. Our "data" is 25 Gaussian blobs on a grid, the same 25 targets as the film's GAN court. Each blob is a **mode**, and a good generator should produce all 25.
#
# A **GAN** (Goodfellow et al., 2014) trains two networks against each other. The generator G turns random noise into samples. The discriminator D tries to tell real samples from G's. G's loss is D's success, so each one's improvement is the other's new problem.

# %%
CENTERS = torch.tensor([(x, y) for x in range(-2, 3) for y in range(-2, 3)], dtype=torch.float32) * 2

def real_batch(n):
    return CENTERS[torch.randint(0, 25, (n,))] + 0.05 * torch.randn(n, 2)

def modes_covered(samples, min_frac=0.01):
    d = torch.cdist(samples, CENTERS); near = d.min(1).values < 0.3
    counts = torch.bincount(d.argmin(1)[near], minlength=25)
    return int((counts > min_frac * len(samples)).sum()), near.float().mean().item()

mlp = lambda i, o: nn.Sequential(nn.Linear(i, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, o))
seed(0)
G, D = mlp(8, 2), mlp(2, 1)
oG = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.9)); oD = torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.9))
bce = nn.BCEWithLogitsLoss(); gan_log = []
for step in range(CFG['gan_steps'] + 1):
    real, fake = real_batch(256), G(torch.randn(256, 8))
    lD = bce(D(real), torch.ones(256, 1)) + bce(D(fake.detach()), torch.zeros(256, 1))
    oD.zero_grad(); lD.backward(); oD.step()
    lG = bce(D(fake), torch.ones(256, 1))                 # G wins when D calls its fakes real
    oG.zero_grad(); lG.backward(); oG.step()
    if step % (CFG['gan_steps'] // 6) == 0:
        with torch.no_grad():
            s = G(torch.randn(2000, 8))
        gan_log.append((step, s, *modes_covered(s)))
        print(f'step {step:5d}  modes {gan_log[-1][2]:2d}/25  on-target {gan_log[-1][3]:.0%}')

fig, axes = plt.subplots(1, len(gan_log), figsize=(2.2 * len(gan_log), 2.4))
for ax, (step, s, m, _) in zip(axes, gan_log):
    ax.scatter(*CENTERS.T, s=40, facecolors='none', edgecolors=GRAPHITE)
    ax.scatter(*s.T, s=1, c=MAG, alpha=0.4); ax.set(xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), title=f'{step} · {m}/25'); ax.axis('off')

# %%
still(122.9)

# %% [markdown]
# GANs are notoriously unstable. Rerun with different seeds and you'll see runs that cover most modes, and runs that **collapse** onto a few: the generator finds a handful of outputs that fool D and stops exploring. Hold on to your mode count for the next comparison.
#
# ## 3.2 Diffusion learned to denoise in a thousand tiny steps
# **Diffusion models** (Sohl-Dickstein 2015; DDPM, Ho et al. 2020) have no adversary. Take a real sample and add a little Gaussian noise, 1,000 times, until it's pure noise. Then train one network on one boring regression task: *given a noisy sample and the step number t, predict the noise that was added*.
#
# $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon \qquad L = \lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2$$
#
# To generate, start from pure noise and subtract the predicted noise one small step at a time. It's a plain regression loss with nothing to collapse, so diffusion tends to cover every mode.

# %%
still(124.9)

# %%
T = 1000
betas = torch.linspace(1e-4, 0.02, T); abar = torch.cumprod(1 - betas, 0)

class EpsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 16, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))
    def forward(self, x, t):
        f = torch.exp(torch.linspace(0, 6, 8)) * t[:, None] / T           # sinusoidal time embedding
        return self.net(torch.cat([x, f.sin(), f.cos()], 1))

seed(0); eps_net = EpsNet(); opt = torch.optim.Adam(eps_net.parameters(), 1e-3)
with timer('DDPM training:'):
    for step in range(CFG['ddpm_steps']):
        x0 = real_batch(512); t = torch.randint(0, T, (512,)); eps = torch.randn_like(x0)
        xt = abar[t, None].sqrt() * x0 + (1 - abar[t, None]).sqrt() * eps
        loss = F.mse_loss(eps_net(xt, t.float()), eps)
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def ddpm_sample(n, keep=(999, 600, 300, 100, 0)):
    x, frames = torch.randn(n, 2), {}
    for t in reversed(range(T)):
        eps = eps_net(x, torch.full((n,), float(t)))
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / (1 - betas[t]).sqrt()
        if t > 0:
            x += betas[t].sqrt() * torch.randn_like(x)
        if t in keep:
            frames[t] = x.clone()
    return frames

frames = ddpm_sample(2000)
fig, axes = plt.subplots(1, len(frames), figsize=(2.2 * len(frames), 2.4))
for ax, (t, s) in zip(axes, frames.items()):
    ax.scatter(*CENTERS.T, s=40, facecolors='none', edgecolors=GRAPHITE); ax.scatter(*s.T, s=1, c=MAG, alpha=0.4)
    ax.set(xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), title=f't = {t}'); ax.axis('off')
print(f'DDPM: modes {modes_covered(frames[0])[0]}/25   vs   GAN: {gan_log[-1][2]}/25')

# %% [markdown]
# The trade-off: a GAN generates in one forward pass, and DDPM needed a thousand. Much of the work since 2020 (DDIM, distillation, flow matching) has gone into cutting that number down.
#
# ## 3.3 The parts list of a modern LLM
# The Transformer you built in Module 2 is the 2017 design. The next four lines of the verse each swap one of its parts. Here is each one in a few lines of code.

# %%
still(129.3)

# %% [markdown]
# **Positions: sinusoids → RoPE.** Attention is order-blind: shuffle the tokens and the weights shuffle with them. The original Transformer *added* a fixed sinusoidal vector to each token to mark its position. **RoPE** (Su et al., 2021) instead *rotates* the query and key vectors by an angle proportional to the position. The dot product of two rotated vectors depends only on the *difference* of their angles, so attention scores depend only on **relative** position. Let's check numerically:

# %%
def rope(x, pos, base=10000.0):
    d = x.shape[-1]; theta = base ** (-torch.arange(0, d, 2) / d)
    ang = pos * theta; c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

seed(0); q, k = torch.randn(64), torch.randn(64)
for m, n in [(5, 2), (105, 102), (1005, 1002), (7, 2)]:
    print(f'q at {m:4d}, k at {n:4d}  (m − n = {m - n}):  q·k = {rope(q, m) @ rope(k, n):+.5f}')

# %% [markdown]
# **Normalization: LayerNorm → RMSNorm.** Deep nets keep activations at a sane scale by normalizing them in every layer. LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm (2019) skips the mean and divides by the root-mean-square. That's cheaper, and it works just as well in practice.
#
# **Activation: ReLU → SwiGLU.** SwiGLU (Shazeer, 2020) multiplies two linear projections, one passed through Swish(x) = x·σ(x): `down(swish(gate(x)) * up(x))`. It's a learned gate inside the MLP.

# %%
x = torch.tensor([3.1, -0.4, 1.2, 0.8, -2.5, 0.3, 1.9, -1.0])
ln = (x - x.mean()) / x.std(unbiased=False); rms = x / x.pow(2).mean().sqrt()
z = torch.linspace(-4, 3, 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
i = np.arange(8)
a1.bar(i - 0.27, x, 0.27, color=GRAPHITE, label='input'); a1.bar(i, ln, 0.27, color=ASH, label='LayerNorm'); a1.bar(i + 0.27, rms, 0.27, color=MAG, label='RMSNorm')
a1.set_title('NORMALIZE 8 ACTIVATIONS'); a1.legend(fontsize=8)
a2.plot(z, torch.relu(z), c=ASH, label='ReLU'); a2.plot(z, z * torch.sigmoid(z), c=MAG, label='Swish (SwiGLU\'s gate)')
a2.set_title('ACTIVATIONS'); a2.legend(fontsize=8);

# %% [markdown]
# These aren't just textbook parts. Download the configuration of this course's LLM (a few KB) and you'll find every one of them:

# %%
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(CFG['llm'])
for k in ['model_type', 'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads', 'intermediate_size',
          'hidden_act', 'rms_norm_eps', 'rope_theta', 'max_position_embeddings', 'vocab_size']:
    print(f'{k:>24}: {getattr(cfg, k, getattr(cfg, "rope_parameters", {}).get(k) if k == "rope_theta" else None)}')

# %% [markdown]
# `hidden_act: silu` is Swish, used inside a SwiGLU MLP. `rms_norm_eps` means RMSNorm. `rope_theta` is RoPE's base frequency. Every block of this model is the verse's parts list.
#
# ## 3.4 Dense became a mixture: only some experts awake
# A dense layer uses every weight for every token. A **mixture of experts** (Shazeer et al., 2017) replaces the MLP with E smaller expert MLPs plus a **router** that sends each token to its top-k experts. You get E× the parameters at about k/E of the compute per token.
#
# Toy version: 8 input clusters, each needing a different function. Can a router with 8 experts and top-2 routing learn to divide the work?

# %%
still(136.0)

# %%
class MoE(nn.Module):
    def __init__(self, d=16, E=8, k=2, hidden=32):
        super().__init__()
        self.k, self.router = k, nn.Linear(d, E)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1)) for _ in range(E))
    def forward(self, x):
        probs = self.router(x).softmax(-1)
        topv, topi = probs.topk(self.k, -1)
        out = torch.zeros(len(x), 1)
        for slot in range(self.k):
            for e, expert in enumerate(self.experts):
                sel = topi[:, slot] == e
                if sel.any():                                   # only routed tokens touch expert e
                    out[sel] += topv[sel, slot, None] * expert(x[sel])
        balance = (probs.mean(0) * F.one_hot(topi[:, 0], len(self.experts)).float().mean(0)).sum() * len(self.experts)
        return out, topi, balance

seed(0)
centers = torch.randn(8, 16) * 3; funcs = torch.randn(8, 16)
def moe_batch(n=512):
    c = torch.randint(0, 8, (n,)); x = centers[c] + torch.randn(n, 16)
    return x, (torch.sin(x * funcs[c]).sum(1, keepdim=True)), c

moe = MoE(); opt = torch.optim.Adam(moe.parameters(), 3e-3)
for step in range(1500):
    x, y, _ = moe_batch(); out, _, bal = moe(x)
    loss = F.mse_loss(out, y) + 0.01 * bal                     # + load-balancing loss keeps all experts in use
    opt.zero_grad(); loss.backward(); opt.step()
x, y, c = moe_batch(4000); _, topi, _ = moe(x)
routing = torch.zeros(8, 8)
for ci in range(8):
    routing[ci] = torch.bincount(topi[c == ci].flatten(), minlength=8).float() / (2 * (c == ci).sum())
fig, ax = plt.subplots(figsize=(4.4, 4)); ax.imshow(routing, cmap='baseline', vmin=0, vmax=1); ax.grid(False)
ax.set(xlabel='expert', ylabel='input cluster', title='WHERE THE ROUTER SENDS EACH CLUSTER')
print(f'final MSE {F.mse_loss(moe(x)[0], y).item():.3f}; each token uses 2/8 experts = 25% of expert compute')

# %% [markdown]
# The router learned a division of labor on its own: each cluster goes mostly to a couple of experts. Without the load-balancing term, routers tend to collapse onto a few favorite experts. Try `0 * bal`.
#
# ## 3.5 Meet a real LLM
# Time to load `CFG['llm']`. A causal language model is the Module 2 TinyGPT with more of everything: it takes tokens and returns a probability for every possible next token. Here is its next-token distribution at the end of the chorus line.

# %%
tok, llm = load_llm()
prompt = "You were state of the art, now you're just a"
ids = tok(prompt, return_tensors='pt').to(DEVICE)
with torch.no_grad():
    p = llm(**ids).logits[0, -1].float().softmax(-1)
top = p.topk(10)
fig, ax = plt.subplots(figsize=(6, 3))
labels = [repr(tok.decode([i])) for i in top.indices.tolist()]
ax.barh(labels, top.values.cpu(), color=[MAG] + [GRAPHITE] * 9); ax.invert_yaxis()
ax.set(title=f'P(NEXT TOKEN | "…just a")', xlabel='probability');

# %% [markdown]
# ## 3.6 32 bits down to 4, for the throughput's sake
# Each parameter is a 32-bit (or 16-bit) float. LLM inference is mostly limited by **memory bandwidth**: generating each token streams every weight from memory once. Store each weight in 4 bits and you move 8× less data, and the model fits on smaller machines.
#
# **Quantization** maps each row of a weight matrix onto 16 evenly spaced levels: q = round(w / scale), clipped to [−8, 7], with scale = max|w| / 7 per row. Below we quantize every linear layer of the real LLM and measure the damage as **perplexity** (e^loss, lower is better) on a passage it hasn't seen.

# %%
still(138.5)

# %%
passage = shakespeare()[-6000:]
eval_ids = tok(passage, return_tensors='pt')['input_ids'][:, :1024].to(DEVICE)

@torch.no_grad()
def perplexity(model):
    return math.exp(model(eval_ids, labels=eval_ids).loss.item())

linears = [m for n, m in llm.named_modules() if isinstance(m, nn.Linear) and 'lm_head' not in n]
originals = [m.weight.data.to('cpu', copy=True) for m in linears]

def fake_quant(w, bits, group=None):
    """Quantize w to signed `bits`-bit integers with one scale per row (or per group of columns), then dequantize."""
    shape = w.shape; w = w.float()
    if group:
        w = w.reshape(-1, group)
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return ((w / scale).round().clamp(-qmax - 1, qmax) * scale).reshape(shape)

results = {'original': perplexity(llm)}
for label, bits, group in [('INT8 per-row', 8, None), ('INT4 per-row', 4, None), ('INT4 group-128', 4, 128), ('INT3 group-128', 3, 128)]:
    for m, w in zip(linears, originals):
        m.weight.data = fake_quant(w.to(DEVICE), bits, group).to(m.weight.dtype)
    results[label] = perplexity(llm)
for m, w in zip(linears, originals):
    m.weight.data = w.to(DEVICE, m.weight.dtype)
del originals

n_lin = sum(m.weight.numel() for m in linears)
orig_bits = next(llm.parameters()).element_size() * 8
for k, v in results.items():
    bits = orig_bits if 'orig' in k else int(k[3])
    print(f'{k:>22}: perplexity {v:7.2f}   linear weights ≈ {n_lin * bits / 8 / 2**20:,.0f} MB at {bits} bits')

# %% [markdown]
# INT8 is nearly free. INT4 per row hurts, because one outlier weight sets the scale for the whole row and crushes everything else into a few levels. Give every **group of 128** weights its own scale and most of the quality comes back for about 3% more storage. That's the core idea behind practical 4-bit formats (GPTQ, AWQ, GGUF's Q4_K, MLX 4-bit). On the `cuda` tier you can compare real 4-bit kernels with `bitsandbytes`.
#
# ## 3.7 Full fine-tune → LoRA: a low-rank little patch
# **Fine-tuning** continues training a pretrained model on your own data. Updating all its weights means storing a full copy of the model per task, plus optimizer state 2–3× its size. **LoRA** (Hu et al., 2021) freezes W and learns a low-rank update ΔW = B·A, where A is r × d and B is d × r with r ≈ 8. Only B and A are trained. That's under 1% of the parameters, and the patch is a few MB.
#
# `add_lora` in `blcourse.py` wraps the attention's `q_proj` and `v_proj` layers. Task: teach the model the film's house style for "baseline cards", a format it has never seen.

# %%
still(143.8)

# %%
CARDS = [
    ('Perceptron', 'PERCEPTRON · 1958 · ROSENBLATT · BEATEN BY: MULTI-LAYER NETS'),
    ('SVM', 'SVM · 1995 · CORTES & VAPNIK · BEATEN BY: DEEP CNNS'),
    ('AlexNet', 'ALEXNET · 2012 · KRIZHEVSKY ET AL. · BEATEN BY: DEEPER CNNS'),
    ('ResNet', 'RESNET · 2015 · HE ET AL. · BEATEN BY: VISION TRANSFORMERS'),
    ('LSTM', 'LSTM · 1997 · HOCHREITER & SCHMIDHUBER · BEATEN BY: TRANSFORMERS'),
    ('Seq2seq', 'SEQ2SEQ · 2014 · SUTSKEVER ET AL. · BEATEN BY: ATTENTION'),
    ('GAN', 'GAN · 2014 · GOODFELLOW ET AL. · BEATEN BY: DIFFUSION'),
    ('BERT', 'BERT · 2018 · DEVLIN ET AL. · BEATEN BY: DECODER-ONLY LLMS'),
    ('Word2vec', 'WORD2VEC · 2013 · MIKOLOV ET AL. · BEATEN BY: CONTEXTUAL EMBEDDINGS'),
    ('Dropout', 'DROPOUT · 2014 · SRIVASTAVA ET AL. · BEATEN BY: MORE DATA'),
    ('Batch normalization', 'BATCH NORMALIZATION · 2015 · IOFFE & SZEGEDY · BEATEN BY: LAYER NORMALIZATION'),
    ('GRU', 'GRU · 2014 · CHO ET AL. · BEATEN BY: TRANSFORMERS'),
    ('VGG', 'VGG · 2014 · SIMONYAN & ZISSERMAN · BEATEN BY: RESNET'),
    ('Adam', 'ADAM · 2014 · KINGMA & BA · BEATEN BY: ADAMW'),
    ('GPT-2', 'GPT-2 · 2019 · RADFORD ET AL. · BEATEN BY: GPT-3'),
    ('Variational autoencoder', 'VARIATIONAL AUTOENCODER · 2013 · KINGMA & WELLING · BEATEN BY: DIFFUSION'),
]
HELD_OUT = ['Transformer', 'Random forest', 'Hidden Markov model', 'U-Net']
Q = 'Baseline card: {}'

print('BEFORE (zero-shot):')
for name in HELD_OUT[:2]:
    print(f'  {name}: {chat(tok, llm, Q.format(name), 40)[:110]!r}')

# %%
lora_layers = add_lora(llm, r=8)
trainable = [p for p in llm.parameters() if p.requires_grad]
print(f'trainable: {sum(p.numel() for p in trainable):,} of {n_params(llm):,} = {sum(p.numel() for p in trainable) / n_params(llm):.3%}')
opt = torch.optim.AdamW(trainable, lr=1e-3)
seed(0); losses = []
with timer('LoRA fine-tune:'):
    for step in range(CFG['lora_steps']):
        batch = [CARDS[i] for i in torch.randint(0, len(CARDS), (4,)).tolist()]
        loss = 0
        for name, card in batch:
            ids, n = chat_ids(tok, Q.format(name), card)
            loss = loss - answer_logprob(llm, ids, n).mean() / len(batch)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
fig, ax = plt.subplots(figsize=(6, 2.6)); ax.plot(losses, c=MAG); ax.set(title='LORA TRAINING LOSS (NATS PER ANSWER)', xlabel='step');

# %%
print('AFTER LoRA, on techniques it never saw in training:')
for name in HELD_OUT:
    print(f'  {name:>20}: {chat(tok, llm, Q.format(name), 40)}')

# %% [markdown]
# The format transferred to techniques that weren't in the 16 examples, and the model fills in the year and authors from what it already knew. The facts can still be wrong, and a small model is often confidently wrong. **LoRA teaches form far more reliably than knowledge.** In real projects, it's the right tool for style, format and tone, and a poor way to add facts. For facts, put them in the prompt (next section) or use retrieval.
#
# ## 3.8 Then just prompt it in context: no gradient to catch
# GPT-3 (2020) showed that a big enough model can pick up a task from a few examples placed in its prompt, with no training at all. This is **in-context learning**. Let's turn the LoRA patch off and try that instead. For a chat model, the examples go in as earlier turns of the conversation.

# %%
still(146.0)

# %%
few_shot = [(Q.format(n), c) for n, c in CARDS[:5]]       # five examples, as earlier turns of the conversation
with lora_disabled(lora_layers):
    for name in HELD_OUT:
        print(f'  {name:>20}: {chat(tok, llm, Q.format(name), 40, history=few_shot)}')

# %%
still(147.8)

# %% [markdown]
# **Developer's rule of thumb:** try prompting first, since it costs nothing and you can change it at any time. Fine-tune with LoRA when the prompt gets too long, too expensive, or too unreliable, or when you need a behavior that's hard to describe with examples.
#
# ### Checkpoint
# 1. What does a GAN's discriminator do, and what is mode collapse? Why doesn't DDPM collapse?
# 2. RoPE makes attention scores depend on what?
# 3. A 7B model in 16-bit takes about 14 GB. How much at INT4 with group scales? Why do group scales help?
# 4. LoRA with r = 8 on a 4096 × 4096 matrix: how many trainable numbers, versus a full fine-tune of that matrix?
# <details><summary>Answers to 3 and 4</summary>
#
# 3: about 14 × 4/16 ≈ 3.5 GB, plus a few % for scales. One outlier no longer sets the step size for a whole row.
# 4: 2 × 8 × 4096 = 65,536 versus 16,777,216, which is 0.39%. Those are the numbers on the film's LoRA plate.
# </details>
#
# **Next:** Module 4, alignment, FlashAttention and long context.
