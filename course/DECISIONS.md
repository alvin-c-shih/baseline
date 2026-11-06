# Course design decisions

A record of what was decided for the *Baseline* ML course, and why, for later review. Newest changes are at the top of each section. "Open" items need a decision.

## Audience and scope

- **Who:** professional software developers who have never taken an ML class. They are assumed to read Python and use a terminal. No calculus or linear algebra is assumed: the math is always shown but always optional.
- **Pacing:** self-paced, chosen over 6 × 2-hour sessions or a one-day workshop. So every module ends with checkpoint questions, and the labs have *Try it* exercises with folded answers.
- **Language:** English. The repo's own docs are in Traditional Chinese; the root README's documents table links to the course with a one-line Chinese description.
- **Order by dependency, anchored to the song** (revised; it was "syllabus = the song"). The first version followed the film's sections, with one deviation (M0 first). A learnability review found three places where film order put a topic before what it depends on, so the modules were reordered. See *Reorder for learnability* below.
- **Additions the film doesn't cover, for this audience:** train/val/test and overfitting, initialization, embeddings, using a pretrained model from a hub, and "always run a baseline" (M5's closing advice, with a table of baselines by problem type).
- **Shrunk:** the Hamming/ECC moment in pre-chorus 2 (MEMORY'S) isn't ML and gets no section.

## Format

- **The module pages are the "slides".** The course is a static HTML site (`site/`), plus one Jupyter notebook per module (`notebooks/`). There is no slide deck file. The user referred to the module pages as slides, and feedback on them is tracked below. *Open:* if an actual presentation deck is wanted for live teaching, generate it from `site/content.py`.
- **Six notebooks instead of one.** The request said "a companion Jupyter notebook". It was split per module because a single notebook would be about 150 cells with roughly 20 minutes of CPU training top to bottom, hard to restart midway. Shared code lives in `notebooks/blcourse.py`.
- **Source of truth:** `notebooks/src/*.py` (jupytext percent format) → `build_notebooks.sh --execute` → `.ipynb`, committed **with the outputs of one `laptop`-tier run** (revised; they were committed empty). The site's LAB links open GitHub's notebook view, where an empty notebook shows no plots and no stills, so a reader couldn't see a lab without installing it. The `laptop` tier is the one most readers can reproduce. Stills are a plain image output plus a separate caption output, because GitHub's viewer drops images embedded in HTML output. Site content lives in `site/content.py` and is rendered by `site/build.py`.

## Reorder for learnability (from the content review)

The review found three problems, fixed together:

**1. Hidden prerequisites.** Module 1 used `F.cross_entropy`, mini-batches, epochs, AdamW, a learning-rate schedule, augmentation and `eval()` without explaining any of them; cross-entropy was first explained in M2. Code from M2 on assumes `(B, T, C)` shapes. M0 now teaches them before anything uses them:
- §0.4 *From numbers to classes*: logits, softmax, cross-entropy, why logs. Trains the §0.0 digits with the five-line loop (94% test).
- §0.5 *The loop at scale*: mini-batch, epoch, shuffle, optimizer (SGD vs Adam), schedule, augmentation, eval/no_grad, hyperparameters, each mapped to a line of M1's CNN loop. Demo: in 15 epochs full-batch SGD reaches test loss 1.33; mini-batches reach about 0.2.
- §0.6 *Vectors, matrices and shapes*: dot product as match score, matmul, axis conventions.
- M2 explains sampling temperature where `generate` first uses it.

**2. Film order vs dependency order.** The modules now are:

| Module | Contents | Was |
|---|---|---|
| M0 The Loop | + classification, the loop at scale, shapes | M0 |
| M1 One Line, Then Many | perceptron → … → HOG → **SVM** → CNN → ResNet (the lab used an SVM before §1.5 taught it) | M1 |
| M2 Sequences | + encoder vs decoder (BERT/GPT), + today's parts list (RoPE, RMSNorm, SwiGLU) | M2, M4.1, M3.3 |
| M3 Generating Data | GAN, diffusion | M3.1–3.2 |
| M4 A Real LLM | meet it, MoE, quantization, LoRA, prompting, RLHF/Bradley–Terry, DPO, on one loaded model | M3.4–3.8, M4.2–4.3 |
| M5 Scale and Long Context | scaling and serving cost, n² and the KV cache, FlashAttention, needle in a haystack, linear time, baselines | M5, M4.4–4.5 |

The cost: a module no longer maps to one continuous stretch of the film. `SONG` in `build.py` now maps each film segment to the module that teaches it; module pages link every stretch they cover, the index timeline shows modules as scattered segments, and the reel assigns each still to the module whose concept uses it. Every concept keeps its own still and timestamp, so the film still orients the reader.

Implementation notes: M4 loads the LLM once; DPO calls `reset_lora` (zero the §4.4 patch) instead of loading a second copy and wrapping LoRA twice. M5 loads the LLM itself for the needle test. The duplicate `bench()` was removed. New site concepts: *From numbers to classes*, *The loop at scale*, *Vectors, matrices and shapes*, *Meet a real LLM*. 10 glossary terms added (114 total).

**3. Too little practice.** Before, M2 and M3 had no exercises and most cells ran without the learner writing code. Now:
- **13 "Your turn" exercises with automatic checks**, two or three per module: numeric gradient, cross-entropy (with the overflow trap), shapes; perceptron update, hand-wired XOR; one attention row, temperature; DDPM forward noising; group quantization, LoRA parameter count, DPO loss; KV-cache bytes, streaming softmax. `check(name, fn)` (in `blcourse.py`) runs the tests in `notebooks/exercises.py` and prints ✓ / ✗ with the failing case / "not attempted", and never stops the notebook. `BASELINE_SOLUTIONS=1` runs the reference solutions through the same tests, so a wrong test or reference fails loudly in course testing.
- **"Predict first" prompts** before every long or surprising cell (about 15).
- **Checkpoint answers**, folded, in every notebook and on every module page.

Not done from the review (priorities 4–5): a core-path toggle for the side boxes, per-module learning objectives, "uses" chips and deep links between site and notebooks, "read more" source links, a course-wide leaderboard, and metrics beyond loss/accuracy.

## Starting before the model (feedback: "take a step back for programmers")

- **New first concept, M0 §0.0 "Why not just write the program?"** Try hand-writing a digit recognizer, see the rules explode, then flip to *a fixed program template with blank numbers plus an algorithm that fills them in*. Why that algorithm must **converge**, and why gradient descent does: a smooth template gives a direction for every blank from one pass.
- **The road not taken:** genetic algorithms and genetic programming (Holland 1975, Koza 1992) and neuroevolution (NEAT 2002) search programs with only "better or worse" feedback, so the number of tries explodes with the number of knobs. Still useful where knobs are few and there is no gradient: NASA's 2006 evolved ST5 antenna, and OpenAI's 2017 evolution-strategies results on many CPUs.
- **Runnable in the notebook:** scikit-learn's bundled 8×8 digits (no download). Six hand-written rules get 36%, never outputting a 2, 4 or 5. A 650-number linear template learned from 70% of the data gets 95% on the other 30%. Its weight images are shown as "the program nobody wrote".
- **"model" defined for programmers:** "a program whose logic you don't write by hand: a fixed template of simple math with millions of blank numbers, filled in automatically from examples."
- **Core terms are always hoverable.** *model, parameter, loss, gradient* are linked at their first mention in every concept, marked or not (`CORE` in `build.py`), so "the model" in a later box (such as M2's "Average surprise") has a definition. 155 links in all, each at its first mention.

## Bridging the leaps (revised after feedback: "leaps of understanding are assumed")

The review found that most concepts leaned on a few unexplained ideas: the dot product as a match score, the chain rule, compounding, a fold with an accumulator, a dictionary lookup, a loop vs a parallel map, the memory hierarchy, caching. The fix is an **ML-agnostic stepping stone** before each explanation:

- **"A familiar idea first"** box (`bridge` in `content.py`) on 38 of 39 concepts, placed right after the one-liner and *before* the body, which now builds on it. Examples: FlashAttention ← loop-tiled matrix multiply + running max/sum + CPU cache vs RAM; RNN ← `reduce` with an accumulator; attention ← a soft dict lookup; Transformer ← a loop vs a parallel map; RoPE ← clock hands; LoRA ← a Docker layer and a low-rank spreadsheet edit; Bradley–Terry ← Elo ratings; KV cache ← memoization; linear-time models ← a fixed-size sketch (HyperLogLog); Chinchilla ← `-O3` for a binary you run a billion times.
- **Rule for writing one:** use something a working developer has already done (written a regex, a fold, a cache, an A/B test), not another ML idea. If a bridge needs its own explanation, it's the wrong bridge.
- **The notebooks get runnable versions where the leap is biggest:** M4 §4.4 now does a tiled matmul and a streaming softmax as separate cells before FlashAttention. The tiled matmul differs from `A @ B` by about 1e-4 because float additions happen in a different order, and the notebook says so.

## Commercial implications (also from that feedback)

- **"The business side"** box (`biz`) on 35 concepts: what an idea did to cost, products or markets. Examples: LeNet reading checks; the $44M DNNresearch auction and Nvidia's CUDA position; Netflix Prize code never deployed; BERT in Google Search; GPT-3 as a metered API; ChatGPT's growth; multi-LoRA serving; quantization enabling local LLMs; MoE and the DeepSeek market reaction; KV-cache memory setting long-context prices; prompt caching.
- **Chinchilla gets the full trade-off.** Compute-optimal for *training* is not optimal for a model you *serve*: training ≈ 6ND, serving ≈ 2N per token. Worked example on the page: Chinchilla-70B (1.4T tokens) and Llama-3-8B (15T tokens) cost about the same to train (5.9 vs 7.2 × 10²³ FLOPs), but the 8B is about 9× cheaper per served token. At 10¹² tokens a day, the 70B's serving compute matches its training cost in about four days. The M5 notebook has a cell for this calculation and plot, with `served_per_day` as the knob.
- Same sourcing rule as Lore: documented facts only, with hedges such as "reported" and "an estimated" where the numbers are estimates.

## Page structure (revised after feedback: "too dense for a newcomer")

Each concept section, in reading order:
1. **Kicker** (notebook section · year) and **title**.
2. **One line:** the idea in one plain sentence, large, with a magenta rule. A newcomer can read only these and still follow the module.
3. **A familiar idea first** (see above).
4. **Body:** two to four sentences, trimmed from the first version, then an optional **worked example**. Technical words are marked `[[term]]` in `content.py`.
5. **The math:** folded behind a `∑ The math` toggle. Equations never block the reading flow.
6. **LAB** tag naming the notebook section.
7. Beside the text (below it on phones): the **film still** with its caption, then three boxes (*Back then*, *The business side*, *Lore*):
   - **Back then:** what people were struggling with in that era, such as untrainable deep nets, finicky GANs, PPO's four models in memory, or attention's memory traffic.
   - **Lore:** memes, stories and jokes from the time, such as the 1958 NYT perceptron hype, STACK MORE LAYERS, the Schmidhuber-invented-it meme, Goodfellow's bar story, "divine benevolence", Sesame Street model names, the shoggoth meme, "go brrr", and the Dolores Park needle.

**Lore rule:** only documented stories. Anecdotes that are folklore, or whose exact wording is disputed, are labeled as such (Jelinek's "fire a linguist" line, "one book caused the AI winter", AlexNet in a bedroom marked "reportedly"). Direct quotes are kept short and attributed.

## Glossary and hover definitions

- 103 terms in `GLOSSARY` (`site/content.py`), each defined in one or two plain sentences. `build.py` fails if a page marks a term that has no entry, so a definition can't go missing.
- **The first visible mention gets the link, wherever the `[[mark]]` sits.** Once bridges came before bodies, 40 terms (including "nats") had their first mention in an unmarked box. Now `[[term]]` only declares that a concept uses a term, and `gloss_in_order` links its first mention in reading order (one-liner → bridge → body → example → boxes). It matches plurals and a small `ALIASES` list (backprop, hidden, kernel, compute…), works on text nodes only, and skips code and existing links. A build check confirmed 109 links with no plain mention before any of them.
- "nats" got a friendlier definition (a unit of surprise; e^loss options), and the M2 notebook now explains nats and perplexity in a sentence, plus an `<abbr>` hover.
- **Behavior:** a dotted underline. With a mouse, hovering or keyboard focus shows a tooltip, positioned to stay on screen. On touch screens the first tap shows the definition and a second tap opens the glossary entry; hover handlers are off there, because touch browsers fire a synthetic `mouseenter` before `click`. Without JS, the native `title` tooltip and the link still work.
- A **Glossary page** lists every term alphabetically with tags for the modules that use it, and is linked from the nav.
- *Open:* the notebooks' markdown has no hover definitions yet. Jupyter would render `<abbr title>` but not this tooltip. Add them if students mostly read inside Jupyter.

## Visual style

- Tokens from `docs/DESIGN.md`: ink `#0B0A0D`, bone `#EDEAF2`, ash/graphite grays, and **magenta `#FF2E7E` only for "now / state of the art"**: the current timecode, the one-line rule. Type: Archivo (wide, 900, uppercase) for headings, IBM Plex Mono for labels and captions.
- The site is dark-only, matching the film.
- **Notebook plots are plain matplotlib** (revised after feedback: "the output graphs should be visually distinct from the video stills"). They first used the film's look: ink background, IBM Plex Mono, uppercase titles, an ink → magenta → bone colormap. In a notebook that put a plot of the student's own result right next to a frame of the film, and the two read as the same kind of thing. Now `blcourse.use_style()` only sets a wide default figure size. Plots use the default color cycle (blue `C0`, orange `C1`), with orange for the winner or the "after", black for decision lines and gray for reference lines. Images use stock colormaps: `gray_r` for digits, `RdBu_r` for signed weights, default `viridis` for heatmaps, `RdYlGn` for found/missed. Titles are sentence case. The film's colors survive only in the caption under each still.
- **Hero images:** module pages put the title first and the still below it. Stills with large lyric text behind the title hurt legibility, which was tried and rejected. The index hero uses the plotter's first shot (t = 5.5 s), which is mostly empty dark space.

## Stills from the film

- 69 stills rendered from the film's own engine (`app/scripts/render.ts stills --samples auto --max-samples 64`), at 1920×1080, saved as 1600-px JPEGs (7.7 MB total). This was chosen over cropping the 380-px contact sheets or extracting frames from YouTube.
- The timestamps were chosen per concept from `docs/SHOTLIST.md`. Rejected and re-rendered because of motion blur or a frame between words: 39.9, 41.0, 65.2, 66.0, 76.5, 116.0, 125.5.
- **Complete phrases (revised after feedback: stills showed "STATE OF T", "EVERY FEATURE CARVED BY").** The engine reveals lyrics word by word, letter by letter, and cuts phrases into chunks, so a frame picked by concept often caught a phrase mid-reveal. 39 of 69 stills changed:
  - **Better pause points, same shot (37 stills).** For each still, the last lyric word sung in its shot (from `data/lyrics.json` and the shot table in `docs/data/structure.json`), at the moment its letters are fully lit (about 62% through the word, after the 0.13 s slam) and before the cut. It stays in the same shot, so the visual the concept is about stays in view. Examples: 47.00 → 47.53 *YOU WERE STATE OF THE ART*; 26.50 → 27.01 *EVERY FEATURE CARVED BY HAND*; 32.50 → 32.76 *THE WIDEST MARGIN*; 143.80 → 144.56 *A LOW-RANK LITTLE PATCH*. Where the scene cuts a line into short chunks that never appear together (*AND CAME / IN / AS THE*), the still shows one complete chunk.
  - **Creative liberty, stills only (6 stills, 4 of them also moved; recorded in `captions.json`).** `fovk`: a wider lens when the phrase runs off the frame (22.12 ×1.3, 140.79 ×1.25, 144.56 ×1.2, 177.13 ×1.15, 202.00 ×1.35, which brings *SCALE IS* into *ALL YOU NEED*). `sing`: lyric words re-timed to have just been sung, when the phrase would otherwise only finish on a motion-blurred frame before a cut (147.80, *TO CATCH*). These hooks live in `app/` (`STILL.fovK` in `_v.ts`, `__mv.sing`/`unsing` in `main.ts`, `--fovk`/`--sing` in `render.ts`). The film never uses them; a frame re-rendered with and without the change differs only by the GPU's run-to-run rounding (max 1/255).
  - `site/render_stills.py` re-renders any or all stills from `captions.json`, liberties included.
- **Dividers in the notebooks** (feedback: "the still heralds the new concept… a horizontal divider to reinforce that"). Every `##` section and every Checkpoint opens with a rule (`---` at the top of the heading's cell). The notebooks' stills play two roles. *Openers* sit under a section heading and introduce it. *Callbacks* are the film's version of a result, placed right after the student's matching plot: 15.60 after the hidden-layer plot, 34.41 after the 3D lift, 36.03 closing §1.6. The rules mark sections, not stills, so callbacks stay inside their section, and a callback that ends one sits just above the next rule. One exception: M0's first still (52.80, the deadline board) opens §0.0, so its rule goes above the still, in a cell of its own. Plain `---` renders everywhere: a thin line in JupyterLab, a thicker bar on GitHub.
- **Captions** live in one file, `site/stills/captions.json`, read by both the notebooks and the site. They name the scene and its on-screen words and **do not quote lyric lines**, because the song and lyrics are not MIT-licensed (`LICENSE-SONG.md`).

## Hardware tiers and models

- One `TIERS` table in `blcourse.py`; every lab reads its sizes from `CFG`. Auto-detection: CUDA with ≥ 12 GB → `cuda`; MPS with ≥ 32 GB → `mac-max`; anything else → `laptop`. Override with `BASELINE_TIER` / `BASELINE_DEVICE`.
- **LLMs: Qwen2.5-Instruct at 0.5B / 1.5B / 7B.** Apache-2.0 at these sizes, chat-tuned, 32K context, and a real modern parts list (RoPE, RMSNorm, SwiGLU, GQA) that M3 points at in its config. The 3B was skipped because of its research-only license.
- **Quantization and LoRA are hand-written** (fake-quant, `LoRALinear`) rather than using bitsandbytes or peft. That makes the mechanism the lesson, avoids extra dependencies, and works on CPU, MPS and CUDA alike.
- **CIFAR-10 comes from the Hugging Face Hub** (`uoft-cs/cifar10`), not torchvision's Toronto server, which ran at about 65 KB/s during testing (about 40 minutes for 170 MB). The Hub copy took seconds.
- Corpus: Tiny Shakespeare (public domain), never the song's lyrics.

## Lab calibrations (each found by running it)

| Lab | Decision | Why |
|---|---|---|
| M0 learning-rate exercise | diverge at `lr=0.4`, not 0.3 | for this data the threshold is 1/E[x²] ≈ 0.37; 0.3 and 0.35 still converge |
| M1 CIFAR, laptop | 20k images × 8 epochs | at 10k × 3 the CNN only tied HOG + SVM (54.8% vs 54.6%); now 68.5% vs 55.8% in about 2 minutes on CPU. `mac-max`: 88.6% vs 56.5% |
| M1 gradient plot | He init for ReLU, Xavier for sigmoid | with PyTorch's default init ReLU also decayed, contradicting the lesson |
| M2 attention maps | last layer, not first | first-layer heads only attend to the previous character |
| M3 in-context learning | few-shot examples as earlier chat turns | pasted as one user message, the chat model ignored the format |
| M4 DPO | rejected = the model's own answers; lr 2e-4 | hand-written "rambling" rejects changed nothing (70 → 70 words); at 5e-4 the answers collapsed into nonsense, which is kept as a *Try it* on over-optimization; at 2e-4, 102 → 40 words and still coherent |
| M5 scaling fit | bounds: 0 ≤ c ≤ min loss, α ≤ 2 | unbounded, the fit found the floor c = −2024; bounded it gives α ≈ 0.10 |

## Bigger tiers: overfitting, speed, downloads (found by the `mac-max` run)

- **The bigger tier overfit and was slow.** On `mac-max`, M2's 10.8M-parameter GPT memorized the 1 MB corpus: validation loss went 1.53 → 3.58, and the run took 33 minutes; the LSTM overfit too. Fixes: (1) `train_lm` now does **early stopping**, keeping and restoring the best-validation checkpoint, which M2 explains as the validation set doing the job Module 0 gave it; (2) **dropout 0.2** and smaller GPTs on `mac-max` (6 × 256, 4.8M parameters) and `cuda` (6 × 384, nanoGPT's Shakespeare setup); (3) TinyGPT now uses PyTorch's fused `scaled_dot_product_attention` instead of `nn.MultiheadAttention`, going from about 0.66 to about 0.16 s per step on MPS. Lesson recorded: a bigger tier needs more *data* to deserve a bigger model. With a fixed corpus, scale the budget carefully.
- DPO keeps 40 steps on every tier. Its learning rate was calibrated at 40 steps, and more steps over-optimized.
- **Hugging Face Xet downloads failed here** (`CAS Client Error`, CDN host unreachable) and left a 43 KB `.incomplete` file, which transformers reported as a misleading "can't load pytorch_model.bin". With `HF_HUB_DISABLE_XET=1` the 2.9 GB downloaded in 3 minutes. `load_llm` now pre-downloads and fails with that advice. The README has a troubleshooting note.

## Verification status

After the reorder, all six notebooks were run top to bottom twice on an M1 Max with 64 GB: on the `laptop` tier forced to CPU in **student mode** (every exercise unattempted, which proves no later cell depends on a student's answer), and on `mac-max` (MPS) with **`BASELINE_SOLUTIONS=1`** (all 13 reference solutions pass their tests inside the notebooks).

| Notebook | `laptop` on CPU, student mode | `mac-max` on MPS, solutions |
|---|---|---|
| M0 | 10 s ✓ | 10 s ✓ (3/3 exercises) |
| M1 | 159 s ✓ (CNN 68.5% vs HOG + SVM 55.8%) | 250 s ✓ (88.6% vs 56.5%; 2/2) |
| M2 | 244 s ✓ (LSTM and GPT tie at 1.54) | 587 s ✓ (GPT 1.49, both early-stopped; 2/2) |
| M3 | 23 s ✓ | 37 s ✓ (1/1) |
| M4 | 282 s ✓ (DPO 102 → 51 words) | 214 s ✓ (100 → 56 words; 3/3) |
| M5 | 130 s ✓ (α ≈ 0.10; needles 20/20 up to 2K) | 319 s ✓ (floor ≈ 1.24; misses needles from 1K up; 2/2) |

**Committed outputs** come from a third run: `laptop` tier on MPS, student mode, 8.0 min in all (M0 9 s, M1 79 s with CNN 68.9% vs 55.8%, M2 161 s, M3 23 s, M4 105 s with DPO 102 → 40 words, M5 76 s with needles 20/20 up to 2K). `blcourse.py` silences torch's macOS-only "Redirects are currently not supported" warning, which otherwise lands in three of them.

DPO's effect varies with the hardware and the LoRA patch's random start (it is reset before DPO): 102 → 51 words on CPU, 102 → 40 on MPS, 100 → 56 on `mac-max`. The site says "roughly halve … to 40–55".

- **`cuda` tier: not run.** No NVIDIA GPU was available. Its sizes are extrapolated, not measured. *Open:* run it once on a 12–24 GB card.
- **Site:** checked at 1440 px and 375 px, with no horizontal scroll on any page, KaTeX rendering, and tooltips tested with mouse hover and touch tap. 155 glossary links, each at its term's first mention. Not tested: a real phone, or screen-reader output of the tooltips.

## Open questions

1. Is an actual presentation deck wanted for live delivery, in addition to the site?
2. Should the notebooks get hover definitions (`<abbr title>`) too?
3. ~~Should the site be published as a hosted page?~~ Resolved: GitHub Pages, from the fork (`.github/workflows/course-pages.yml`).
4. Who runs the `cuda` tier check?
