# Baseline: ML for software developers

A self-paced first course in machine learning for professional developers who have never taken one. It uses the *Baseline* music video as its syllabus. The film names more than thirty techniques in historical order, from the 1958 perceptron to attention and its linear-time successors. Each course module pairs one section of the film with the ideas behind it, anchored by stills rendered from the film's own engine, plus a Jupyter lab where you build those ideas at a size your machine can run.

| | |
|---|---|
| **Read** | `site/`: static pages in the film's palette. Open `site/index.html`, or serve the folder: `python3 -m http.server -d course` then visit `/site/`. |
| **Run** | `notebooks/`: one notebook per module, with shared code in `notebooks/blcourse.py`. |
| **Watch** | https://youtu.be/Qy3_4oXoS6c. Every still's timecode links to that moment. |

## Modules

| | Film | Topics | Lab highlights |
|---|---|---|---|
| **M0 · The Loop** | chorus · 0:45–1:00 | model, loss, gradient descent, autograd, train/val/test, overfitting, benchmarks | gradient descent by hand; an overfitting curve |
| **M1 · One Line, Then Many** | verse 1 + DEPTH · 0:00–0:45 | perceptron, XOR, backprop, vanishing gradients, ReLU and init, HOG vs CNN, SVM and kernels, residual connections | backprop in numpy (the film's 2-2-1 network, seed 7); HOG + SVM vs a CNN on CIFAR-10; a 30-layer net with and without residuals |
| **M2 · Sequences** | verse 2 + MEMORY + GONE · 1:00–1:59 | tokens/BPE, n-grams, RNN, BPTT, LSTM, seq2seq, attention, Transformer | n-gram → LSTM → TinyGPT on one leaderboard; seq2seq ± attention reversing digit strings |
| **M3 · The Lab** | verse 3 · 1:59–2:28 | GAN, diffusion, RoPE, RMSNorm, SwiGLU, MoE, quantization, LoRA, in-context learning | GAN vs DDPM on the film's 25 modes; INT8/INT4/group-128 quantization of a real LLM; LoRA vs few-shot |
| **M4 · Bridge** | chorus 3 + bridge · 2:43–3:19 | BERT vs decoder-only, RLHF, Bradley–Terry, DPO, FlashAttention, long context | DPO on a real LLM; FlashAttention's online softmax from scratch; a needle-in-a-haystack map |
| **M5 · Scale** | SCALE → end · 3:19–4:07 | scaling laws, O(n²) attention, KV cache, linear attention/SSMs, baselines | fit your own scaling law; chunked linear attention vs softmax attention |

## Setup

Needs Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
cd course
uv sync
uv run jupyter lab notebooks/
```

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

Each notebook was run top to bottom with nbconvert on an M1 Max with 64 GB. The `laptop` column was forced onto the CPU to show the slowest case; a laptop's GPU is faster. The `cuda` tier has not been run yet.

| Notebook | `laptop` on CPU | `mac-max` on MPS |
|---|---|---|
| `00_the_loop` | 11 s | 9 s |
| `01_one_line_then_many` | 3.5 min (CNN 68.5% vs HOG 55.8%) | 6.3 min (CNN 88.6% vs HOG 56.5%) |
| `02_sequences` | 4.0 min (LSTM 1.54, GPT 1.54) | 9.9 min (LSTM 1.51, GPT 1.47) |
| `03_the_lab` | 1.9 min | 3.1 min |
| `04_alignment_and_scale_out` | 3.6 min | 3.7 min |
| `05_scale_and_footnotes` | 1.7 min | 2.9 min |

The first run of M1, M3 and M4 also includes the downloads.

## Editing

- **Notebooks**: edit `notebooks/src/*.py` (jupytext percent format), then run `sh build_notebooks.sh` to regenerate `notebooks/*.ipynb` without outputs. `--execute` also runs each notebook as a test, writing the executed copies to `.cache/executed/`.
- **Site**: edit the modules, concepts and glossary in `site/content.py` (layout lives in `site/build.py`), then run `python3 site/build.py`. Mark glossary terms with `[[term]]` or `[[term|shown text]]`.
- **Design decisions** and open questions: `DECISIONS.md`.
- **Stills**: `site/stills/tSSS.ss.jpg` are 1600-px frames rendered from the film's engine with `cd app && bun scripts/render.ts stills --samples auto --max-samples 64 --t <times>`. Their captions live in `site/stills/captions.json`, which both the notebooks and the site read.

## License

Course text and code are MIT, like the rest of the repo. The song *Baseline*, its recording and its lyrics are **not** MIT (see `../LICENSE-SONG.md`). The course describes scenes and their on-screen words; it doesn't reproduce the lyrics. Qwen2.5 0.5B, 1.5B and 7B, and DistilBERT, are Apache-2.0. Tiny Shakespeare is public domain. CIFAR-10 is from Krizhevsky (2009).
