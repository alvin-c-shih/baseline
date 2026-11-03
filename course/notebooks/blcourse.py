"""Shared helpers for the Baseline course notebooks.

    from blcourse import *          # TIER, CFG, DEVICE, colors, still(), seed(), timer(), ...

Hardware tiers. Every lab reads its sizes from CFG, so one switch scales the whole course:

    laptop   any modern laptop, CPU or Apple-silicon GPU. Every lab finishes in minutes.
    mac-max  Apple M-series Max/Ultra with >= 32 GB unified memory (MPS backend).
    cuda     an NVIDIA GPU with >= 12 GB VRAM.

The tier is auto-detected. Override it before importing, or from the shell:

    BASELINE_TIER=laptop  BASELINE_DEVICE=cpu  jupyter lab
"""
from __future__ import annotations

import contextlib
import math
import os
import platform
import random
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
REPO = COURSE.parent
STILLS = COURSE / 'site' / 'stills'
FONTS = REPO / 'app' / 'public' / 'fonts'
CACHE = Path(os.environ.get('BASELINE_CACHE', COURSE / '.cache'))
CACHE.mkdir(parents=True, exist_ok=True)

# ── the film's palette (docs/DESIGN.md §2). Magenta is reserved for "the current state of the art". ──
INK, INK2, PANEL, LINE = '#0B0A0D', '#121016', '#17151C', '#2A2731'
GRAPHITE, ASH, BONE = '#5C5966', '#9A96A3', '#EDEAF2'
MAG, MAG_HOT, MAG_DEEP, MAG_DARK = '#FF2E7E', '#FF9CC4', '#A3104A', '#3A0A1E'


# ── hardware ────────────────────────────────────────────────────────────────────────────────────────
def _ram_gb() -> float:
    try:
        if platform.system() == 'Darwin':
            return int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'])) / 2**30
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
    except Exception:
        return 8.0


def _detect_device() -> str:
    want = os.environ.get('BASELINE_DEVICE')
    if want:
        return want
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _detect_tier(device: str) -> str:
    want = os.environ.get('BASELINE_TIER')
    if want:
        return want
    if device == 'cuda' and torch.cuda.get_device_properties(0).total_memory / 2**30 >= 12:
        return 'cuda'
    if device == 'mps' and _ram_gb() >= 32:
        return 'mac-max'
    return 'laptop'


DEVICE = _detect_device()
TIER = _detect_tier(DEVICE)

# One row per tier. A lab that needs a size looks it up here and nowhere else.
TIERS = {
    'laptop': dict(
        cifar_train=20_000, cifar_epochs=8, cnn_width=32, resnet=False,
        text_chars=1_000_000,
        gpt=dict(n_layer=4, n_head=4, n_embd=128, block=128, steps=1500, batch=32),
        lstm=dict(hidden=256, steps=1500, batch=32, block=128),
        scaling_widths=[32, 48, 64, 96, 128], scaling_steps=600,
        gan_steps=3000, ddpm_steps=4000,
        llm='Qwen/Qwen2.5-0.5B-Instruct', llm_dtype='float32',
        lora_steps=60, dpo_steps=40,
        niah_lengths=[256, 512, 1024, 2048], niah_depths=[0.0, 0.25, 0.5, 0.75, 1.0],
        attn_lengths=[256, 512, 1024, 2048],
    ),
    'mac-max': dict(
        cifar_train=50_000, cifar_epochs=10, cnn_width=64, resnet=True,
        text_chars=None,
        # Tiny Shakespeare is only ~1 MB: bigger nets memorize it, so the bigger tiers add dropout and keep the best
        # validation checkpoint (train_lm) rather than just training longer
        gpt=dict(n_layer=6, n_head=4, n_embd=256, block=256, steps=2000, batch=64, dropout=0.2),
        lstm=dict(hidden=512, steps=2000, batch=64, block=256),
        scaling_widths=[32, 64, 96, 128, 192, 256], scaling_steps=1500,
        gan_steps=6000, ddpm_steps=8000,
        llm='Qwen/Qwen2.5-1.5B-Instruct', llm_dtype='bfloat16',
        lora_steps=150, dpo_steps=40,    # DPO's lr and steps were calibrated on laptop; more steps over-optimizes
        niah_lengths=[512, 1024, 2048, 4096, 8192, 16384], niah_depths=[0.0, 0.25, 0.5, 0.75, 1.0],
        attn_lengths=[256, 512, 1024, 2048, 4096, 8192],
    ),
    'cuda': dict(
        cifar_train=50_000, cifar_epochs=20, cnn_width=64, resnet=True,
        text_chars=None,
        gpt=dict(n_layer=6, n_head=6, n_embd=384, block=256, steps=5000, batch=64, dropout=0.2),   # nanoGPT's Shakespeare setup
        lstm=dict(hidden=512, steps=3000, batch=64, block=256),
        scaling_widths=[32, 64, 128, 192, 256, 384, 512], scaling_steps=3000,
        gan_steps=10_000, ddpm_steps=15_000,
        llm='Qwen/Qwen2.5-7B-Instruct', llm_dtype='bfloat16',
        lora_steps=200, dpo_steps=40,
        niah_lengths=[1024, 2048, 4096, 8192, 16384, 32768], niah_depths=[0.0, 0.25, 0.5, 0.75, 1.0],
        attn_lengths=[512, 1024, 2048, 4096, 8192, 16384],
    ),
}
CFG = TIERS[TIER]


def hardware_report() -> None:
    """Print what this machine is, which tier it gets, and how to override it."""
    print(f'python {platform.python_version()} · torch {torch.__version__} · {platform.machine()} · {_ram_gb():.0f} GB RAM')
    if DEVICE == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f'GPU: {p.name}, {p.total_memory / 2**30:.0f} GB')
    print(f'DEVICE = {DEVICE!r}   TIER = {TIER!r}   (override: BASELINE_TIER / BASELINE_DEVICE)')


# ── reproducibility & timing ───────────────────────────────────────────────────────────────────────
def seed(s: int = 0) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@contextlib.contextmanager
def timer(label: str = ''):
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    print(f'{label} {time.perf_counter() - t0:.2f}s'.strip())


def sync() -> None:
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elif DEVICE == 'mps':
        torch.mps.synchronize()


# ── style ──────────────────────────────────────────────────────────────────────────────────────────
def use_style() -> None:
    """Matplotlib in the film's look: ink background, bone lines, magenta only where we point at something."""
    import matplotlib as mpl
    from matplotlib import font_manager

    for f in ['IBMPlexMono-Regular.ttf', 'IBMPlexMono-Medium.ttf', 'IBMPlexMono-Bold.ttf']:
        p = FONTS / 'src' / f
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({
        'figure.facecolor': INK, 'axes.facecolor': INK, 'savefig.facecolor': INK,
        'axes.edgecolor': LINE, 'axes.labelcolor': ASH, 'axes.titlecolor': BONE,
        'axes.grid': True, 'grid.color': LINE, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': ASH, 'ytick.color': ASH, 'text.color': BONE,
        'font.family': ['IBM Plex Mono', 'DejaVu Sans Mono'], 'font.size': 10,
        'axes.titleweight': 'bold', 'axes.titlesize': 11, 'axes.titlelocation': 'left',
        'legend.frameon': False, 'legend.labelcolor': BONE,
        'axes.prop_cycle': mpl.cycler(color=[BONE, ASH, GRAPHITE, MAG_HOT, MAG_DEEP]),
        'image.cmap': 'baseline', 'figure.figsize': (7, 3.6), 'figure.dpi': 110,
        'lines.linewidth': 1.6, 'scatter.edgecolors': 'none',
    })
    from matplotlib.colors import LinearSegmentedColormap
    for name, stops in {'baseline': [INK, MAG_DARK, MAG_DEEP, MAG, MAG_HOT, BONE],
                        'bone': [INK, GRAPHITE, ASH, BONE]}.items():
        if name not in mpl.colormaps:
            mpl.colormaps.register(LinearSegmentedColormap.from_list(name, stops))


# ── stills from the film ───────────────────────────────────────────────────────────────────────────
def still(t: float, caption: str | None = None, width: int = 720):
    """Show the film's frame at song time t (seconds), captioned from site/stills/captions.json: where you are in the video."""
    from IPython.display import HTML, display
    import base64
    import json
    p = STILLS / f't{t:06.2f}.jpg'
    if not p.exists():
        print(f'(still {p.name} not found)')
        return
    info = json.loads((STILLS / 'captions.json').read_text()).get(f'{t:.2f}', {})
    caption = caption or f"{info.get('scene', '')} · {info.get('c', '')}"
    b = base64.b64encode(p.read_bytes()).decode()
    m, s = divmod(t, 60)
    display(HTML(
        f'<figure style="margin:0 0 8px 0;max-width:{width}px">'
        f'<img src="data:image/jpeg;base64,{b}" style="width:100%;border:1px solid {LINE}"/>'
        f'<figcaption style="font:12px \'IBM Plex Mono\',monospace;color:{ASH};margin-top:4px">'
        f'<a href="https://youtu.be/Qy3_4oXoS6c?t={int(t)}" style="color:{MAG}">{int(m)}:{s:05.2f}</a> · {caption}</figcaption></figure>'))


# ── downloads ──────────────────────────────────────────────────────────────────────────────────────
def fetch(url: str, name: str) -> Path:
    """Download url into the course cache once; return the local path."""
    p = CACHE / name
    if not p.exists():
        print(f'downloading {url} → {p}')
        urllib.request.urlretrieve(url, p)
    return p


SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def shakespeare() -> str:
    """Tiny Shakespeare (public domain, ~1.1 MB): the course's text corpus."""
    text = fetch(SHAKESPEARE_URL, 'tinyshakespeare.txt').read_text()
    return text[:CFG['text_chars']] if CFG['text_chars'] else text


CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def cifar10():
    """CIFAR-10 as uint8 arrays: (train_x N×32×32×3, train_y, test_x, test_y). Downloaded once from the
    Hugging Face Hub (uoft-cs/cifar10, ~140 MB), then cached as .npz."""
    p = CACHE / 'cifar10.npz'
    if not p.exists():
        import io
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from PIL import Image
        arrays = {}
        for split in ('train', 'test'):
            f = hf_hub_download('uoft-cs/cifar10', f'plain_text/{split}-00000-of-00001.parquet', repo_type='dataset')
            t = pq.read_table(f)
            arrays[f'{split}_x'] = np.stack([np.asarray(Image.open(io.BytesIO(d['bytes'])).convert('RGB'))
                                             for d in t.column('img').to_pylist()])
            arrays[f'{split}_y'] = np.asarray(t.column('label').to_pylist())
        np.savez(p, **arrays)
    d = np.load(p)
    return d['train_x'], d['train_y'], d['test_x'], d['test_y']


class CharData:
    """Character-level tokenizer + 90/10 train/val split + random batches."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.train, self.val = data[:n], data[n:]

    @property
    def vocab(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return ''.join(self.chars[i] for i in ids)

    def batch(self, split: str, block: int, batch: int, device: str = DEVICE):
        d = self.train if split == 'train' else self.val
        ix = torch.randint(len(d) - block - 1, (batch,))
        x = torch.stack([d[i:i + block] for i in ix])
        y = torch.stack([d[i + 1:i + block + 1] for i in ix])
        return x.to(device), y.to(device)


# ── a tiny GPT (Modules 2, 3, 5) ───────────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)      # after the projection, not inside the fused kernel (MPS has no fused dropout)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def heads(self, x):
        """Split into per-head queries, keys, values: each (B, heads, T, head_dim)."""
        B, T, C = x.shape
        return [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in self.qkv(x).split(C, dim=-1)]

    def attn_weights(self, x):
        """The causal attention pattern of each head, (B, heads, T, T), for plotting. Training uses the fused kernel."""
        q, k, _ = self.heads(self.ln1(x))
        T = q.shape[-2]
        s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        return s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), -float('inf')).softmax(-1)

    def forward(self, x):
        q, k, v = self.heads(self.ln1(x))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(a.transpose(1, 2).flatten(2)))          # residual: the DEPTH lesson
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Decoder-only transformer: token + position embeddings, causal self-attention blocks, a linear head."""

    def __init__(self, vocab: int, n_layer: int, n_head: int, n_embd: int, block: int, dropout: float = 0.0, **_):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, dropout) for _ in range(n_layer))
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)

    def embed(self, idx):
        return self.drop(self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device)))

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n: int, temperature: float = 0.8):
        self.eval()                          # no dropout while sampling
        for _ in range(n):
            logits, _ = self(idx[:, -self.block:])
            p = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(p, 1)], dim=1)
        return idx


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def eval_loss(model, data: CharData, block: int, iters: int = 20, batch: int = 32) -> dict:
    model.eval()
    out = {}
    for split in ('train', 'val'):
        out[split] = float(np.mean([model(*data.batch(split, block, batch))[1].item() for _ in range(iters)]))
    model.train()
    return out


def train_lm(model, data: CharData, steps: int, block: int, batch: int, lr: float = 3e-3,
             log_every: int = 250, label: str = '', keep_best: bool = True) -> list[tuple[int, float, float]]:
    """AdamW with a one-cycle schedule. Returns [(step, train_loss, val_loss)] at each log point.
    keep_best: at the end, restore the weights from the log point with the lowest validation loss (early stopping),
    so a model that starts memorizing the training text is rolled back to before it did."""
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    hist = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = data.batch('train', block, batch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            e = eval_loss(model, data, block)
            hist.append((step, e['train'], e['val']))
            print(f'{label} step {step:5d}  train {e["train"]:.3f}  val {e["val"]:.3f}  ({time.perf_counter() - t0:.0f}s)')
            if keep_best and e['val'] <= min(h[2] for h in hist):
                best = (step, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if keep_best and best[0] != steps:
        model.load_state_dict(best[1])
        print(f'{label} validation loss was lowest at step {best[0]}: restored those weights (early stopping)')
    return hist


# ── a pretrained LLM from the Hugging Face Hub (Modules 3, 4) ──────────────────────────────────────
def load_llm(name: str | None = None):
    """Load CFG['llm'] (or name) and its tokenizer onto DEVICE. First call downloads the weights to ~/.cache/huggingface."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = name or CFG['llm']
    try:    # download first, so a broken download fails here with advice instead of as a confusing load error
        snapshot_download(name, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
    except Exception as e:
        raise RuntimeError(
            f"Downloading {name} didn't finish ({type(e).__name__}: {str(e)[:160]}).\n"
            "Rerun this cell; downloads resume where they stopped. If it keeps failing with a 'CAS' or 'xet' error, "
            "your network is blocking Hugging Face's Xet CDN: restart Jupyter with HF_HUB_DISABLE_XET=1. "
            "Or use a smaller tier: BASELINE_TIER=laptop.") from e
    dtype = getattr(torch, CFG['llm_dtype']) if DEVICE != 'cpu' else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEVICE).eval()
    print(f'{name}: {n_params(model) / 1e6:,.0f}M parameters, {dtype}, on {DEVICE}')
    return tok, model


def chat(tok, model, prompt: str, max_new_tokens: int = 60, system: str | None = None,
         history: list[tuple[str, str]] = ()) -> str:
    """One greedy chat turn. history = earlier (user, assistant) turns, e.g. few-shot examples."""
    msgs = [{'role': 'system', 'content': system}] if system else []
    for u, a in history:
        msgs += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': a}]
    msgs.append({'role': 'user', 'content': prompt})
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()


class LoRALinear(nn.Module):
    """W x + (alpha/r) · B A x, with W frozen and only A (r × in) and B (out × r) trained. B starts at zero,
    so the wrapped layer begins exactly equal to the original."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / r, True
        for p in base.parameters():
            p.requires_grad_(False)
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev, dtype=dt) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))

    def forward(self, x):
        y = self.base(x)
        return y + (x @ self.A.T @ self.B.T) * self.scale if self.enabled else y


def add_lora(model, targets=('q_proj', 'v_proj'), r: int = 8, alpha: float = 16) -> list[LoRALinear]:
    """Freeze model and wrap every Linear whose name ends in one of targets. Returns the LoRA layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                lora = LoRALinear(child, r, alpha)
                setattr(mod, child_name, lora)
                layers.append(lora)
    return layers


@contextlib.contextmanager
def lora_disabled(layers):
    """Temporarily run the frozen base model (the DPO reference policy)."""
    for l in layers:
        l.enabled = False
    try:
        yield
    finally:
        for l in layers:
            l.enabled = True


def chat_ids(tok, prompt: str, answer: str | None = None):
    """Token ids of a chat prompt (and answer). Returns (ids, n_prompt_tokens)."""
    msgs = [{'role': 'user', 'content': prompt}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pi = tok(p, add_special_tokens=False)['input_ids']
    if answer is None:
        return torch.tensor([pi]), len(pi)
    ai = tok(answer + tok.eos_token, add_special_tokens=False)['input_ids']
    return torch.tensor([pi + ai]), len(pi)


def answer_logprob(model, ids, n_prompt: int):
    """Sum of log p(answer tokens | prompt) under model: the quantity both SFT and DPO push on."""
    ids = ids.to(DEVICE)
    logits = model(ids).logits[:, :-1].float()
    lp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]
    return lp[:, n_prompt - 1:].sum(-1)


def check(name: str, attempt) -> None:
    """Run exercise `name`'s tests (exercises.py) against your attempt and say how it went. Never raises, so the
    notebook keeps running. With BASELINE_SOLUTIONS=1 the reference solution is tested instead (course self-test)."""
    import exercises
    test = exercises.TESTS[name]
    if os.environ.get('BASELINE_SOLUTIONS') == '1':
        test(exercises.SOLUTIONS[name])             # a wrong test or reference must fail loudly here
        print(f'✓ {name}: reference solution passes (BASELINE_SOLUTIONS=1)')
        return
    try:
        test(attempt)
    except NotImplementedError:
        print(f'⏳ {name}: not attempted yet. Fill in the code above and rerun this cell. '
              f'(Stuck? The reference solution is in notebooks/exercises.py.)')
    except AssertionError as e:
        print(f'✗ {name}: not yet. {e}')
    except Exception as e:
        print(f'✗ {name}: your code raised {type(e).__name__}: {e}')
    else:
        print(f'✓ {name}: passed.')


def reset_lora(layers) -> None:
    """Zero every LoRA patch (B = 0) and re-draw A, so the model is back to its base weights with a fresh patch."""
    with torch.no_grad():
        for l in layers:
            l.B.zero_()
            l.A.normal_(0, 1 / math.sqrt(l.A.shape[1]))


use_style()

__all__ = [
    'TIER', 'TIERS', 'CFG', 'DEVICE', 'CACHE', 'hardware_report', 'seed', 'timer', 'sync', 'use_style', 'still', 'fetch',
    'shakespeare', 'cifar10', 'CIFAR10_CLASSES', 'CharData', 'TinyGPT', 'n_params', 'eval_loss', 'train_lm', 'load_llm', 'chat',
    'LoRALinear', 'add_lora', 'lora_disabled', 'reset_lora', 'chat_ids', 'answer_logprob', 'check',
    'INK', 'INK2', 'PANEL', 'LINE', 'GRAPHITE', 'ASH', 'BONE', 'MAG', 'MAG_HOT', 'MAG_DEEP', 'MAG_DARK',
    'np', 'torch', 'nn', 'F', 'math', 'time',
]
