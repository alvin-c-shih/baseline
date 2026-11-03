# %% [markdown]
# # Module 0 · The Loop
# **Baseline: machine learning for software developers**
#
# Every technique in the song was trained the same way: show the model examples, score how wrong it is, nudge the numbers to be less wrong, and repeat. This notebook builds that loop by hand, then lets PyTorch do the calculus.
#
# You will learn: **model, parameters, loss, gradient descent, autograd, train/validation/test splits, overfitting, benchmarks.**
#
# > Developer translation: a *model* is a function with tunable constants. *Training* is an automated search for constants that make a test suite pass. The *loss* is that test suite, made differentiable so the search knows which way to move.

# %%
import sys; sys.path.insert(0, '.')
from blcourse import *
import matplotlib.pyplot as plt
hardware_report()

# %% [markdown]
# ### Setup: the hardware switch
# Every lab in the course reads its sizes from one table, `CFG`. Your tier was detected above:
#
# | tier | for | what changes |
# |---|---|---|
# | `laptop` | any modern laptop | small models, data subsets; every lab finishes in minutes |
# | `mac-max` | M-series Max/Ultra, ≥ 32 GB | wider/deeper nets, full datasets, a 1.5B LLM |
# | `cuda` | NVIDIA GPU, ≥ 12 GB | biggest configs, a 7B LLM |
#
# To force a tier, restart the kernel with `BASELINE_TIER=laptop jupyter lab` (or set `os.environ['BASELINE_TIER']` before the import).

# %%
CFG

# %%
still(52.8)

# %% [markdown]
# ## 0.0 Why not just write the program?
# Before any ML, try the obvious approach. scikit-learn ships 1,797 small handwritten digits (8 × 8 pixels, values 0–16; no download needed). Your job: write `classify(img)` by hand. Below is a first attempt with six rules a programmer might write. Run it, then try to improve it.

# %%
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()
imgs, labels = digits.images, digits.target

fig, axes = plt.subplots(1, 10, figsize=(10, 1.3))
for d, ax in enumerate(axes):
    ax.imshow(imgs[np.where(labels == d)[0][0]], cmap='bone'); ax.set_title(str(d), fontsize=9); ax.axis('off')

def classify(img):
    """Hand-written rules. Each one is reasonable; together they still fail."""
    ink = img.sum()
    center = img[3:5, 3:5].sum()
    top, bottom = img[:3].sum(), img[5:].sum()
    left, right = img[:, :4].sum(), img[:, 4:].sum()
    if center < 10 and ink > 250: return 0            # a ring with an empty middle
    if img[:, 3:5].sum() > 0.55 * ink: return 1        # mostly one vertical stroke
    if top > 1.6 * bottom: return 7                    # heavy on top
    if bottom > 1.4 * top: return 6                    # heavy at the bottom
    if right > 1.5 * left: return 9
    return 8 if ink > 330 else 3

rule_pred = np.array([classify(i) for i in imgs])
print(f'hand-written rules: {(rule_pred == labels).mean():.0%} correct')
print('per digit:', '  '.join(f'{d}:{(rule_pred[labels == d] == d).mean():.0%}' for d in range(10)))

# %% [markdown]
# Every rule makes sense, and the program still misses most digits: it never even outputs a 2, 4 or 5. Fix one digit and you break another. Now flip the approach. Keep a fixed *template*: give each of the 10 digits a score equal to a weighted sum of the 64 pixels, and pick the highest score. That leaves 64 × 10 weights + 10 biases = 650 blank numbers. Then let an algorithm fill in the blanks from labeled examples. We train on 70% of the digits and test on the other 30%, which it never saw.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(imgs.reshape(len(imgs), -1), labels, test_size=0.3, random_state=0)
learned = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
print(f'learned template: {learned.score(Xte, yte):.0%} correct on digits it never saw')

fig, axes = plt.subplots(1, 10, figsize=(10, 1.6))
for d, ax in enumerate(axes):
    w = learned.coef_[d].reshape(8, 8); lim = np.abs(w).max()
    ax.imshow(w, cmap='baseline', vmin=-lim, vmax=lim); ax.set_title(str(d), fontsize=9); ax.axis('off')
fig.suptitle("THE 650 BLANKS, FILLED IN · EACH DIGIT'S PIXEL WEIGHTS (BRIGHT = VOTES FOR, DARK = AGAINST)",
             x=0.01, y=1.04, ha='left', fontsize=9, color=BONE);

# %% [markdown]
# Those ten little images *are* the program. Nobody wrote them; an algorithm found them by following the gradient of the error, which is the subject of the rest of this module.
#
# **The road not taken.** You could also search for programs by evolution: mutate candidates, keep the fittest (genetic algorithms, genetic programming). That works for a handful of knobs, but each try only says "better or worse", so the number of tries explodes as knobs grow. The gradient gives a direction for all 650 knobs (or 175 billion) from one pass over the data.
#
# ### Try it
# Spend five minutes improving `classify`. How far can you get? Then imagine doing it for 28 × 28 photos of street signs.

# %% [markdown]
# ## 0.1 A model is a function with knobs
# Here is the smallest useful model: a line, ŷ = w·x + b, with two **parameters** w and b. The data is noisy points from a line we pretend not to know.

# %%
seed(0)
x = np.random.uniform(-3, 3, 60)
y = 1.7 * x - 0.4 + np.random.normal(0, 0.8, x.shape)

fig, ax = plt.subplots()
ax.scatter(x, y, s=14, c=ASH)
ax.set(title='DATA · 60 POINTS', xlabel='x', ylabel='y');

# %% [markdown]
# ## 0.2 The loss is a test suite that returns a number
# A unit test passes or fails. A **loss** tells you *how badly* the model fails, as one smooth number. Mean squared error is the classic choice:
#
# $$L(w, b) = \frac{1}{N}\sum_i (w x_i + b - y_i)^2$$
#
# Because L is smooth, its **gradient** ∂L/∂w, ∂L/∂b points uphill. Step the other way and the loss goes down. That step, repeated, is **gradient descent**:
#
# $$w \leftarrow w - \eta \,\frac{\partial L}{\partial w}$$
#
# η (eta) is the **learning rate**: the step size.

# %%
def loss_and_grads(w, b):
    err = w * x + b - y
    return (err ** 2).mean(), (2 * err * x).mean(), (2 * err).mean()   # derived by hand

w, b, lr = 0.0, 0.0, 0.05
trace = []
for step in range(60):
    L, dw, db = loss_and_grads(w, b)
    trace.append((w, b, L))
    w, b = w - lr * dw, b - lr * db
print(f'learned w={w:.3f} b={b:.3f}   (truth: 1.7, -0.4)')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
a1.plot([t[2] for t in trace], c=BONE); a1.set(title='LOSS PER STEP', xlabel='step', yscale='log')
xs = np.linspace(-3, 3, 2)
for i in (0, 3, 10):
    a2.plot(xs, trace[i][0] * xs + trace[i][1], c=GRAPHITE, lw=1)
a2.scatter(x, y, s=10, c=ASH); a2.plot(xs, w * xs + b, c=MAG, lw=2.5, label='after 60 steps')
a2.set(title='THE LINE, LEARNING'); a2.legend();

# %% [markdown]
# ### Try it
# Change `lr` to `0.2`, then `0.4`, and rerun. What happens to the loss curve? (Too-large learning rates are the most common reason training "explodes".)
#
# <details><summary>Answer</summary>
#
# At 0.2 it converges faster. At 0.4 each step overshoots the minimum by more than it corrects, so w oscillates with growing amplitude and the loss climbs to infinity. The safe step size depends on the curvature of the loss. That is why optimizers like Adam adapt it per parameter.
# </details>

# %% [markdown]
# ## 0.3 Autograd: never derive a gradient by hand again
# Deriving `dw` by hand works for two parameters. GPT-3 has 175 billion. PyTorch records every operation you do on a tensor marked `requires_grad=True`, then `loss.backward()` walks that record backwards with the chain rule (**backpropagation**, Module 1) and fills in `.grad`.

# %%
X, Y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
w = torch.zeros((), requires_grad=True)
b = torch.zeros((), requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.05)          # the update rule above, packaged
for step in range(60):
    loss = ((w * X + b - Y) ** 2).mean()
    opt.zero_grad()
    loss.backward()                              # fills w.grad, b.grad
    opt.step()
print(f'autograd: w={w.item():.3f} b={b.item():.3f}')

# %% [markdown]
# That five-line loop (**forward → loss → zero_grad → backward → step**) is the heartbeat of every model in this course, from the 1958 perceptron to a 7B LLM. Only the model and the loss change.

# %% [markdown]
# ## 0.4 Train, validation, test: don't grade your own homework
# A model can do perfectly on its training examples and still be useless, just as code can pass every test it was tuned against and still fail in production. So we split the data:
#
# - **train**: the model learns from this.
# - **validation**: you pick settings (model size, learning rate, when to stop) by looking at this.
# - **test**: you look at this *once*, at the end, to report the number.
#
# Below, polynomials of increasing degree fit the same 20 noisy points. Watch the training error keep falling while validation error turns around. That turn is **overfitting**: the model has memorized the noise.

# %%
seed(1)
def make(n):
    x = np.random.uniform(-1, 1, n)
    return x, np.sin(3 * x) + np.random.normal(0, 0.25, n)
xtr, ytr = make(20); xva, yva = make(200)

degrees = range(1, 16)
tr_err, va_err, fits = [], [], {}
for d in degrees:
    coef = np.polyfit(xtr, ytr, d)
    tr_err.append(np.mean((np.polyval(coef, xtr) - ytr) ** 2))
    va_err.append(np.mean((np.polyval(coef, xva) - yva) ** 2))
    fits[d] = coef

best = degrees[int(np.argmin(va_err))]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
a1.plot(degrees, tr_err, c=ASH, label='train'); a1.plot(degrees, va_err, c=BONE, label='validation')
a1.axvline(best, c=MAG, ls='--', lw=1); a1.set(yscale='log', xlabel='polynomial degree', title='ERROR VS. CAPACITY'); a1.legend()
xs = np.linspace(-1, 1, 300)
a2.scatter(xtr, ytr, c=ASH, s=16)
a2.plot(xs, np.polyval(fits[15], xs), c=GRAPHITE, label='degree 15 (memorized)')
a2.plot(xs, np.polyval(fits[best], xs), c=MAG, lw=2.5, label=f'degree {best} (best on validation)')
a2.set(ylim=(-2, 2), title='TWO FITS'); a2.legend(fontsize=8);

# %% [markdown]
# ## 0.5 Benchmarks and leaderboards
# The chorus is about **benchmarks**: a fixed public test set and a metric that everyone reports. ImageNet top-5 error, WMT14 BLEU, LAMBADA accuracy. Benchmarks are what let a field say "state of the art". They also have the weakness the song is about. A benchmark rewards whatever wins on it, and when everyone tunes against the same test set, it slowly turns into a validation set.

# %%
still(47.0)

# %% [markdown]
# ### Checkpoint
# You can now explain, in your own words:
# 1. What a parameter, a loss, and a gradient are.
# 2. The five-line training loop.
# 3. Why we need a validation set separate from the test set.
#
# **Next:** Module 1, where the model stops being a line.
