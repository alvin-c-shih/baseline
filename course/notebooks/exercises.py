"""Exercise tests and reference solutions for the Baseline course.

In a notebook:

    def numeric_grad(f, w):
        raise NotImplementedError      # your code here
    check('numeric_grad', numeric_grad)

check() runs the tests below against your function and prints ✓, ✗ (with the failing case) or "not attempted".
It never stops the notebook. The reference solutions are at the bottom of each entry. Try first, then peek.

BASELINE_SOLUTIONS=1 makes check() run each reference solution through its tests instead (used to test the course).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

TESTS, SOLUTIONS = {}, {}


def exercise(name):
    def deco(test):
        TESTS[name] = test
        return test
    return deco


def solution(name):
    def deco(fn):
        SOLUTIONS[name] = fn
        return fn
    return deco


def close(a, b, tol=1e-5, what='value'):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, f'{what}: shape {a.shape}, expected {b.shape}'
    assert np.allclose(a, b, atol=tol, rtol=tol), f'{what}: got {np.round(a, 5).tolist()}, expected {np.round(b, 5).tolist()}'


# ── Module 0 ───────────────────────────────────────────────────────────────────────────────────────
@exercise('numeric_grad')
def _(fn):
    cases = [(lambda w: (w - 3) ** 2, 1.0, -4.0, '(w − 3)² at w = 1'),
             (math.sin, 0.5, math.cos(0.5), 'sin(w) at w = 0.5'),
             (lambda w: math.exp(2 * w), 0.0, 2.0, 'exp(2w) at w = 0')]
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=20), rng.normal(size=20)
    wt = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    ((wt * torch.tensor(x) - torch.tensor(y)) ** 2).mean().backward()
    cases.append((lambda w: float(np.mean((w * x - y) ** 2)), 0.7, wt.grad.item(), 'the line\'s MSE loss at w = 0.7 (vs autograd)'))
    for f, w, expected, what in cases:
        close(fn(f, w), expected, tol=1e-4, what=f'slope of {what}')


@solution('numeric_grad')
def _(f, w, eps=1e-5):
    return (f(w + eps) - f(w - eps)) / (2 * eps)     # rise over run, measured on both sides of w


@exercise('cross_entropy')
def _(fn):
    cases = [([2.0, 1.0, 0.1], 0), ([0.0, 0.0, 0.0, 0.0], 3), ([1.0, 5.0, -2.0], 2), ([1000.0, 999.0, 0.0], 1)]
    for scores, label in cases:
        expected = F.cross_entropy(torch.tensor([scores]), torch.tensor([label])).item()
        got = float(fn(np.array(scores), label))
        assert math.isfinite(got), f'scores={scores}: got {got}. Huge scores overflow exp(); subtract the max score first.'
        close(got, expected, tol=1e-4, what=f'cross_entropy({scores}, {label})')


@solution('cross_entropy')
def _(scores, label):
    z = scores - scores.max()                       # same softmax, but exp() can't overflow
    return -(z[label] - np.log(np.exp(z).sum()))


SHAPE_QUESTIONS = {
    'a': ('x @ W                     x: (32, 784), W: (784, 10)', lambda: torch.zeros(32, 784) @ torch.zeros(784, 10)),
    'b': ('imgs.flatten(1)           imgs: (32, 3, 32, 32)', lambda: torch.zeros(32, 3, 32, 32).flatten(1)),
    'c': ('emb(tokens)               tokens: (8, 128) integers, emb = nn.Embedding(65, 64)',
          lambda: torch.nn.Embedding(65, 64)(torch.zeros(8, 128, dtype=torch.long))),
    'd': ('q @ k.transpose(-1, -2)   q, k: (8, 4, 128, 16)', lambda: torch.zeros(8, 4, 128, 16) @ torch.zeros(8, 4, 128, 16).transpose(-1, -2)),
    'e': ('logits[:, -1]             logits: (8, 128, 65)', lambda: torch.zeros(8, 128, 65)[:, -1]),
}


@exercise('shapes')
def _(answers):
    if answers is None or any(answers.get(k) is None for k in SHAPE_QUESTIONS):
        raise NotImplementedError
    for k, (q, make) in SHAPE_QUESTIONS.items():
        got, exp = tuple(answers[k]), tuple(make().shape)
        assert got == exp, f'({k}) {q.split("  ")[0]}: you wrote {got}, the actual shape is {exp}'


SOLUTIONS['shapes'] = {k: tuple(make().shape) for k, (q, make) in SHAPE_QUESTIONS.items()}


# ── Module 1 ───────────────────────────────────────────────────────────────────────────────────────
_X4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


@exercise('perceptron_step')
def _(fn):
    for name, y in [('AND', [0, 0, 0, 1]), ('OR', [0, 1, 1, 1])]:
        w, b = np.zeros(2), 0.0
        for _ in range(25):
            for xi, yi in zip(_X4, y):
                w, b = fn(w, b, xi, yi, 0.1)
        errors = sum(float(w @ xi + b > 0) != yi for xi, yi in zip(_X4, y))
        assert errors == 0, f'after 25 epochs on {name}, {errors}/4 still wrong (w={np.round(w, 2)}, b={b:.2f})'
    w, b = fn(np.array([0.0, 0.0]), 0.0, np.array([1.0, 1.0]), 1.0, 0.1)
    close(w, [0.1, 0.1], what='one update from w=0 on a missed positive (1,1)')


@solution('perceptron_step')
def _(w, b, x, y, lr):
    yhat = float(w @ x + b > 0)
    return w + lr * (y - yhat) * x, b + lr * (y - yhat)


@exercise('xor_weights')
def _(fn):
    W1, b1, w2, b2 = (np.asarray(a, dtype=float) for a in fn())
    assert W1.shape == (2, 2) and b1.shape == (2,) and w2.shape == (2,), 'W1 is 2×2, b1 and w2 have 2 entries'
    step = lambda z: (z > 0).astype(float)
    out = step(step(_X4 @ W1.T + b1) @ w2 + b2)
    close(out, [0, 1, 1, 0], what='XOR outputs for (0,0), (0,1), (1,0), (1,1)')


@solution('xor_weights')
def _():
    W1 = [[1, 1], [-1, -1]]          # h1 = OR:  x1 + x2 > 0.5      h2 = NAND: -x1 - x2 > -1.5
    b1 = [-0.5, 1.5]
    w2, b2 = [1, 1], -1.5             # out = AND(h1, h2)
    return W1, b1, w2, b2


# ── Module 2 ───────────────────────────────────────────────────────────────────────────────────────
@exercise('attention_row')
def _(fn):
    g = torch.Generator().manual_seed(0)
    for T, d in [(5, 4), (12, 8)]:
        q, K, V = torch.randn(d, generator=g), torch.randn(T, d, generator=g), torch.randn(T, 3, generator=g)
        expected = F.scaled_dot_product_attention(q[None, None], K[None], V[None])[0, 0]
        close(fn(q, K, V), expected, tol=1e-5, what=f'attention for one query over {T} keys')


@solution('attention_row')
def _(q, K, V):
    scores = K @ q / math.sqrt(q.shape[-1])         # one similarity score per key
    weights = scores.softmax(-1)                     # percentages that sum to 1
    return weights @ V                               # the blend of the values


@exercise('temperature')
def _(fn):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
    close(fn(logits, 1.0), logits.softmax(-1), what='T = 1 is a plain softmax')
    cold = fn(logits, 0.05)
    assert cold[0] > 0.99, f'T = 0.05 should put almost all probability on the top token, got {cold.tolist()}'
    hot, warm = fn(logits, 5.0), fn(logits, 1.0)
    ent = lambda p: -(p * p.log()).sum().item()
    assert ent(hot) > ent(warm), 'a higher temperature should make the distribution flatter (higher entropy)'
    close(fn(logits, 2.0).sum(), 1.0, what='probabilities sum to 1')


@solution('temperature')
def _(logits, T):
    return (logits / T).softmax(-1)


# ── Module 3 ───────────────────────────────────────────────────────────────────────────────────────
@exercise('noisy')
def _(fn):
    abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 1000), 0)
    g = torch.Generator().manual_seed(0)
    x0, eps = torch.randn(4, 2, generator=g), torch.randn(4, 2, generator=g)
    for t in (0, 500, 999):
        expected = abar[t].sqrt() * x0 + (1 - abar[t]).sqrt() * eps
        close(fn(x0, t, eps, abar), expected, tol=1e-5, what=f'x_t at t={t}')
    x = fn(torch.ones(20000, 2), 999, torch.randn(20000, 2, generator=g), abar)
    assert abs(x.std().item() - 1) < 0.05, 'by t = 999 the sample should be almost pure noise (std ≈ 1)'


@solution('noisy')
def _(x0, t, eps, abar):
    return abar[t].sqrt() * x0 + (1 - abar[t]).sqrt() * eps


# ── Module 4 ───────────────────────────────────────────────────────────────────────────────────────
@exercise('lora_params')
def _(fn):
    for (d_out, d_in, r), n in [((4096, 4096, 8), 65_536), ((896, 896, 8), 14_336), ((4864, 896, 16), 92_160)]:
        assert fn(d_out, d_in, r) == n, f'lora_params({d_out}, {d_in}, {r}) = {fn(d_out, d_in, r)}, expected {n}'


@solution('lora_params')
def _(d_out, d_in, r):
    return r * d_in + d_out * r                      # A is r × d_in, B is d_out × r


@exercise('quantize_groups')
def _(fn):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(16, 256, generator=g)
    w[3, 7] = 40.0                                   # one outlier in row 3
    q = fn(w, 4, 128)
    assert q.shape == w.shape, f'shape {tuple(q.shape)}, expected {tuple(w.shape)}'
    for row in range(16):
        for grp in range(2):
            levels = q[row, grp * 128:(grp + 1) * 128].unique().numel()
            assert levels <= 16, f'row {row}, group {grp}: {levels} distinct values; 4 bits allow at most 16'
    per_row = (w / (w.abs().amax(1, keepdim=True) / 7)).round().clamp(-8, 7) * (w.abs().amax(1, keepdim=True) / 7)
    err_q, err_row = (q - w)[3, 128:].abs().mean(), (per_row - w)[3, 128:].abs().mean()
    assert err_q < err_row / 3, ('the group without the outlier should keep its precision '
                                 f'(mean error {err_q:.3f} vs {err_row:.3f} with one scale per row)')


@solution('quantize_groups')
def _(w, bits, group):
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(-1, group)
    scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return ((g / scale).round().clamp(-qmax - 1, qmax) * scale).reshape(w.shape)


@exercise('dpo_loss')
def _(fn):
    close(fn(-10.0, -12.0, -10.0, -12.0, 0.1), math.log(2), what='policy == reference: the loss is log 2')
    a = fn(-8.0, -14.0, -10.0, -12.0, 0.1)           # chosen got likelier, rejected less likely
    b = fn(-12.0, -10.0, -10.0, -12.0, 0.1)          # the reverse
    assert a < math.log(2) < b, f'moving toward the chosen answer should lower the loss (got {a:.3f} and {b:.3f})'
    close(a, -math.log(1 / (1 + math.exp(-0.1 * ((-8 + 10) - (-14 + 12))))), what='the DPO formula')


@solution('dpo_loss')
def _(pol_w, pol_l, ref_w, ref_l, beta):
    margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
    return -math.log(1 / (1 + math.exp(-margin)))   # -log σ(margin)


# ── Module 5 ───────────────────────────────────────────────────────────────────────────────────────
@exercise('kv_cache_bytes')
def _(fn):
    assert fn(32, 8, 128, 8192, 2) == 2 ** 30, 'a Llama-3-8B-like config at 8,192 tokens in bf16 is exactly 1 GiB'
    assert fn(24, 2, 64, 32768, 2) == 402_653_184, 'Qwen2.5-0.5B-like config at 32K tokens'
    assert fn(1, 1, 1, 1, 1) == 2, 'remember there are two tensors per layer: K and V'


@solution('kv_cache_bytes')
def _(layers, kv_heads, head_dim, n_tokens, bytes_per):
    return 2 * layers * kv_heads * head_dim * n_tokens * bytes_per


@exercise('stream_softmax')
def _(fn):
    g = torch.Generator().manual_seed(0)
    for scale in (1.0, 30.0):
        x = torch.randn(5000, generator=g) * scale
        m, l = fn(x.split(700))
        close(m, x.max().item(), tol=1e-4, what='running max')
        close(l, torch.exp(x - x.max()).sum().item(), tol=1e-3, what='running sum of exp(x − max)')


@solution('stream_softmax')
def _(chunks):
    m, l = -float('inf'), 0.0
    for c in chunks:
        m_new = max(m, c.max().item())
        l = l * math.exp(m - m_new) + torch.exp(c - m_new).sum().item()
        m = m_new
    return m, l
