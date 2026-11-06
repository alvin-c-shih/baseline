# Baseline: ML for software developers

A self-paced first course in machine learning for professional developers who have never taken one. It teaches the ideas in the *Baseline* music video. The film names more than thirty techniques in historical order, from the 1958 perceptron to attention and its linear-time successors. Each course module pairs one section of the film with the ideas behind it, anchored by stills rendered from the film's own engine, plus a Jupyter lab where you build those ideas at a size your machine can run.

| | |
|---|---|
| **Read** | `site/`: static pages in the film's palette. Open `site/index.html`, or serve the folder: `python3 -m http.server -d course` then visit `/site/`. |
| **Run** | `notebooks/`: one notebook per module, with shared code in `notebooks/blcourse.py`. |
| **Watch** | https://youtu.be/Qy3_4oXoS6c. Every still's timecode links to that moment. |

## Modules

The film goes in historical order; the modules go in the order the ideas depend on each other, so some modules cover several stretches of the film. Every concept still links to the exact moment the film shows it.

| | Film | Topics | Lab highlights |
|---|---|---|---|
| **M0 · The Loop** | chorus 1 | why learn programs; model, loss, gradient descent, autograd; classification (softmax, cross-entropy); mini-batches, epochs, optimizers; vectors and tensor shapes; train/val/test; benchmarks | hand-written digit rules (36%) vs a learned template (95%); mini-batch vs full-batch training |
| **M1 · One Line, Then Many** | verse 1 + DEPTH | perceptron, XOR, backprop, vanishing gradients, ReLU and init, HOG, SVM and kernels, CNNs, residual connections | backprop in numpy (the film's 2-2-1 network, seed 7); HOG + SVM vs a CNN on CIFAR-10; 30 layers with and without residuals |
| **M2 · Sequences** | verse 2, MEMORY, GONE, BERT, parts of verse 3 | tokens/BPE, n-grams, RNN, BPTT, LSTM, seq2seq, attention, Transformer, sampling temperature, BERT vs decoder-only, RoPE, RMSNorm, SwiGLU | n-gram → LSTM → TinyGPT on one leaderboard; seq2seq ± attention; DistilBERT fills the film's mask |
| **M3 · Generating Data** | verse 3, first lines | GANs, mode collapse, diffusion | GAN vs DDPM on the film's 25 modes |
| **M4 · A Real LLM** | verse 3, chorus 3, bridge (RLHF, DPO) | a real LLM's outputs, mixture of experts, quantization, LoRA, in-context learning, RLHF and Bradley–Terry, DPO | INT8/INT4/group-128 quantization; LoRA vs few-shot; DPO, all on one Qwen2.5 model |
| **M5 · Scale and Long Context** | bridge (FlashAttention, the tape), SCALE → end | scaling laws, Chinchilla vs serving cost, O(n²) attention, KV cache, FlashAttention, long context, linear attention, baselines | your own scaling law; FlashAttention's tiled online softmax; a needle-in-a-haystack map; linear vs softmax attention |

### How each lab works

- **Predict first:** before each long or surprising cell, a prompt asks you to guess the result.
- **Your turn:** 13 small exercises (two or three per module) where you fill in a function and run `check(...)`. It prints ✓, or ✗ with the failing case, or "not attempted", and never stops the notebook. Reference solutions are in `notebooks/exercises.py`.
- **Checkpoint:** questions at the end of each module, with folded answers (in the notebooks and on the site).

## Setup

Needs Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
cd course
uv sync
uv run jupyter lab notebooks/
```

The notebooks in the repo already hold the outputs of one `laptop`-tier run, plots and stills included, so you can read them on GitHub before installing anything. Your own numbers will differ a little.

The first runs download the following:

| What | Size | Used in |
|---|---|---|
| Tiny Shakespeare | 1 MB | M2, M5 |
| CIFAR-10, from the Hugging Face Hub | ≈ 140 MB | M1 |
| DistilBERT | ≈ 260 MB | M4 |
| The tier's LLM | 1–15 GB | M3, M4 |

Datasets are cached in `course/.cache/`, models in `~/.cache/huggingface/`.

### Hardware tiers

`blcourse.py` detects the hardware and picks a tier. Every lab reads its sizes from `CFG = TIERS[TIER]`, so one table controls the whole course:

| Tier | Detected when | LLM | Notable sizes |
|---|---|---|---|
| `laptop` | anything else (CPU, or a base Apple-silicon GPU) | Qwen2.5-0.5B-Instruct | CIFAR subset of 20k, 4-layer/128-wide GPT, haystacks up to 2K tokens |
| `mac-max` | Apple MPS with ≥ 32 GB unified memory | Qwen2.5-1.5B-Instruct (bf16) | full CIFAR-10 with residual blocks, 6-layer/384-wide GPT, haystacks up to 16K |
| `cuda` | NVIDIA GPU with ≥ 12 GB VRAM | Qwen2.5-7B-Instruct (bf16) | 8-layer/512-wide GPT, haystacks up to 32K |

Override either setting from the shell:

```sh
BASELINE_TIER=laptop BASELINE_DEVICE=cpu uv run jupyter lab notebooks/
```

### Troubleshooting

- **The LLM download stops partway, or fails with a "CAS" or "xet" error.** Some networks block Hugging Face's Xet CDN. Restart with plain HTTPS downloads: `HF_HUB_DISABLE_XET=1 uv run jupyter lab notebooks/`. Partial downloads resume.
- **Out of memory on the LLM labs.** Drop a tier: `BASELINE_TIER=laptop`.

## Measured run times

Each notebook was run top to bottom with nbconvert on an M1 Max with 64 GB. The first `laptop` column was forced onto the CPU to show the slowest case. The second used the GPU, and its outputs are the ones committed. The `cuda` tier has not been run yet.

| Notebook | `laptop` on CPU (student mode) | `laptop` on MPS (student mode, committed) | `mac-max` on MPS (reference solutions) |
|---|---|---|---|
| `00_the_loop` | 10 s | 9 s | 10 s |
| `01_one_line_then_many` | 2.7 min (CNN 68.5% vs HOG + SVM 55.8%) | 1.3 min (CNN 68.9% vs 55.8%) | 4.2 min (CNN 88.6% vs 56.5%) |
| `02_sequences` | 4.1 min | 2.7 min | 9.8 min (GPT beats LSTM; both early-stopped) |
| `03_generating_data` | 23 s | 23 s | 37 s |
| `04_a_real_llm` | 4.7 min (DPO: 102 → 51 words) | 1.8 min (DPO: 102 → 40 words) | 3.6 min (DPO: 100 → 56 words) |
| `05_scale_and_long_context` | 2.2 min (needles 20/20 up to 2K) | 1.3 min (needles 20/20 up to 2K) | 5.3 min (misses needles from 1K up) |

*Student mode* leaves every exercise unattempted, which proves no later cell depends on a student's answer. `BASELINE_SOLUTIONS=1` runs each exercise's reference solution through its tests instead. The first runs of M1, M2, M4 and M5 also include downloads.

## Editing

- **Notebooks**: edit `notebooks/src/*.py` (jupytext percent format), then run `sh build_notebooks.sh --execute` to regenerate `notebooks/*.ipynb` and run each one top to bottom, which refreshes the committed outputs and doubles as a test (about 8 minutes on an M1 Max's GPU). It uses the `laptop` tier unless `BASELINE_TIER` says otherwise. Without `--execute` the notebooks are updated from their sources and every code cell whose code didn't change keeps its output, which is enough after editing only the prose. After changing code, use `--execute`, since outputs further down may depend on it.
- **Site**: edit the modules, concepts and glossary in `site/content.py` (layout lives in `site/build.py`), then run `python3 site/build.py`. Mark glossary terms with `[[term]]` or `[[term|shown text]]`.
- **Design decisions** and open questions: `DECISIONS.md`.
- **Stills**: `site/stills/tSSS.ss.jpg` are 1600-px frames rendered from the film's engine. Re-render any of them with `python3 site/render_stills.py [times…]`, which reads `site/stills/captions.json`, including the two still-only liberties some entries use: `fovk` (a wider lens) and `sing` (a phrase re-timed to have just been sung). Their captions live in `site/stills/captions.json`, which both the notebooks and the site read.

## License

Course text and code are MIT, like the rest of the repo. The song *Baseline*, its recording and its lyrics are **not** MIT (see `../LICENSE-SONG.md`). The course describes scenes and their on-screen words; it doesn't reproduce the lyrics. Qwen2.5 0.5B, 1.5B and 7B, and DistilBERT, are Apache-2.0. Tiny Shakespeare is public domain. CIFAR-10 is from Krizhevsky (2009).
