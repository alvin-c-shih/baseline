"""Build the Baseline course site: index.html, m0.html … m5.html, reel.html.

    python3 course/site/build.py

Pages are static HTML in the film's palette (style.css). Stills and their captions come from stills/ and
stills/captions.json, the same captions the notebooks show. Equations are TeX, rendered in the browser by KaTeX.
"""
from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

SITE = Path(__file__).resolve().parent
CAPS = json.loads((SITE / 'stills' / 'captions.json').read_text())
VIDEO_ID = 'Qy3_4oXoS6c'
VIDEO = f'https://youtu.be/{VIDEO_ID}'
# where the LAB links point: the notebooks folder next to the site locally; the Pages workflow points them at
# GitHub's rendered notebook view instead
NB = os.environ.get('BASELINE_NOTEBOOK_URL', '../notebooks')

# ── the song, as the course reads it: (start s, end s, label, module, chorus?) ──
# Modules follow what depends on what, not film order, so one module can cover several stretches of the film.
SONG = [
    (0.00, 38.91, 'VERSE 1', 1, False), (38.91, 45.81, 'DEPTH', 1, False), (45.81, 60.53, 'CHORUS', 0, True),
    (60.53, 90.37, 'VERSE 2', 2, False), (90.37, 97.24, 'MEMORY', 2, False), (97.24, 119.64, 'CHORUS · GONE', 2, True),
    (119.64, 127.02, 'GAN · DIFFUSION', 3, False), (127.02, 133.72, 'PARTS', 2, False), (133.72, 148.32, 'EFFICIENCY', 4, False),
    (148.32, 155.58, 'ATTN', 2, False), (155.58, 163.28, 'CHORUS', 4, True), (163.28, 170.52, 'BERT', 2, True),
    (170.52, 177.17, 'RLHF · DPO', 4, False), (177.17, 198.98, 'FLASH · TAPE', 5, False), (198.98, 206.18, 'SCALE', 5, False),
    (206.18, 220.58, 'CHORUS', 5, True), (220.58, 247.20, 'FOOTNOTE · FALL', 5, False),
]
LENGTH = 247.2


def spans(n: int) -> list[tuple[float, float]]:
    """The stretches of the film that module n covers, with adjacent segments merged."""
    out = []
    for a, b, _, mod, _ in SONG:
        if mod == n:
            if out and abs(out[-1][1] - a) < 1e-6:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
    return out


def module_at(t: float) -> int:
    """Which module explains the still at time t: the one whose concept uses it, else the song map."""
    for m in MODULES:
        if any(abs(c['t'] - t) < 1e-6 for c in m['concepts']) or abs(m['hero'] - t) < 1e-6:
            return m['n']
    return next(mod for a, b, _, mod, _ in SONG if a <= t < b)


def tc(t: float) -> str:
    m, s = divmod(t, 60)
    return f'{int(m)}:{int(s):02d}'


def still(t: float, big: bool = False) -> str:
    k = f'{t:.2f}'
    c = CAPS[k]
    return (f'<figure class="still"><img src="stills/t{t:06.2f}.jpg" alt="{html.escape(c["scene"])}: {html.escape(c["c"])}" '
            f'loading="{"eager" if big else "lazy"}" width="1600" height="900">'
            f'<figcaption><a href="{VIDEO}?t={int(t)}" target="_blank" rel="noopener">{tc(t)}</a> · '
            f'<span class="scene">{html.escape(c["scene"])}</span> · {html.escape(c["c"])}</figcaption></figure>')


# ── content: modules, concepts, glossary (content.py) ─────────────────────────────────────────────────
from content import ALIASES, GLOSSARY, MODULES  # noqa: E402

GLOSS = {k.lower(): v for k, v in GLOSSARY.items()}
CORE = ('model', 'parameter', 'loss', 'gradient')
TERM = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')


def term_link(key: str, shown: str) -> str:
    d = GLOSS.get(key.lower())
    if d is None:
        raise KeyError(f'glossary has no entry for [[{key}]]')
    slug = re.sub(r'[^a-z0-9]+', '-', key.lower()).strip('-')
    return f'<a class="term" href="glossary.html#{slug}" title="{html.escape(d)}">{shown}</a>'


def gloss(text: str) -> str:
    """[[term]] / [[term|shown]] → a link to the glossary entry whose definition shows on hover or tap (see TIP_JS)."""
    return TERM.sub(lambda m: term_link(m.group(1).strip(), (m.group(2) or m.group(1)).strip()), text)


def gloss_in_order(fields: list[str]) -> list[str]:
    """Link each term a concept marks with [[…]] at its FIRST visible mention, in reading order across the concept's
    fields, wherever the mark itself sits. Later mentions stay plain text. Matches the key, the shown text, ALIASES
    and plurals, in text nodes only (never inside tags, links or code)."""
    wanted: dict[str, set[str]] = {}
    for f in fields:
        for key, shown in TERM.findall(f):
            k = key.strip().lower()
            if k not in GLOSS:
                raise KeyError(f'glossary has no entry for [[{key}]]')
            wanted.setdefault(k, set()).update({k, (shown or key).strip().lower(), *ALIASES.get(k, ())})
    for k in CORE:      # foundational words are defined at their first mention everywhere, marked or not
        wanted.setdefault(k, set()).update({k, *ALIASES.get(k, ())})
    patterns = {k: re.compile(r'(?<![\w-])(?:' + '|'.join(re.escape(x) for x in sorted(forms, key=len, reverse=True))
                              + r')(?:s|es)?(?![\w-])', re.I) for k, forms in wanted.items()}
    links: list[str] = []
    out = []
    for f in (TERM.sub(lambda m: (m.group(2) or m.group(1)).strip(), f) for f in fields):
        parts, skip = re.split(r'(<[^>]+>)', f), 0
        for i, p in enumerate(parts):
            if p.startswith('<'):
                skip += 1 if re.match(r'<(a|code)\b', p) else -1 if re.match(r'</(a|code)>', p) else 0
                continue
            if skip:
                continue
            for k, pat in patterns.items():
                m = pat.search(p) if k not in {l[0] for l in links} else None
                if m:   # swap the match for a placeholder so later terms can't match inside it
                    links.append((k, term_link(k, m.group(0))))
                    p = p[:m.start()] + f'\0{len(links) - 1}\0' + p[m.end():]
            parts[i] = p
        out.append(''.join(parts))
    return [re.sub(r'\0(\d+)\0', lambda m: links[int(m.group(1))][1], o) for o in out]


# ── page shell ─────────────────────────────────────────────────────────────────────────────────────
HEAD = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100..125,400..900&family=IBM+Plex+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}}]}})"></script>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="bar">
  <a class="brand" href="index.html">BASE<b>LINE</b> · ML</a>
  <nav>{nav}</nav>
</header>
<main>
'''
TIP_JS = '''<script>
/* Glossary terms: hover or keyboard focus shows the definition; on touch screens the first tap shows it and a
   second tap opens the glossary entry. Without JS the native title tooltip and the link still work. */
(() => {
  const tip = document.createElement('div'); tip.id = 'tip'; tip.setAttribute('role', 'tooltip'); document.body.append(tip);
  const touch = matchMedia('(hover: none)').matches; let cur = null;
  const show = el => {
    cur = el; tip.textContent = el.dataset.def; tip.style.maxWidth = Math.min(300, innerWidth - 24) + 'px'; tip.classList.add('on');
    el.setAttribute('aria-describedby', 'tip');
    const r = el.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
    const x = Math.max(12, Math.min(r.left + r.width / 2 - tw / 2, innerWidth - tw - 12));
    let y = r.top - th - 8; if (y < 52) y = r.bottom + 8;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  const hide = () => { if (cur) cur.removeAttribute('aria-describedby'); cur = null; tip.classList.remove('on'); };
  for (const el of document.querySelectorAll('.term')) {
    el.dataset.def = el.title; el.removeAttribute('title');
    if (!touch) {    // touch browsers fire a synthetic mouseenter before click, so hover handlers stay off there
      el.addEventListener('mouseenter', () => show(el)); el.addEventListener('mouseleave', hide);
      el.addEventListener('focus', () => show(el)); el.addEventListener('blur', hide);
    }
    el.addEventListener('click', e => { if (touch && cur !== el) { e.preventDefault(); show(el); } });
  }
  addEventListener('scroll', hide, { passive: true });
  addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
  document.addEventListener('click', e => { if (cur && !e.target.closest('.term')) hide(); });
})();
</script>
'''
FOOT = '''</main>
<footer>
  A first machine-learning course built on the music video <a href="{video}">Baseline</a>. Stills are rendered from the film's engine (<code>app/</code>).
  Course text and code: MIT. The song and its lyrics are not covered by the MIT license; see <code>LICENSE-SONG.md</code>.
</footer>
{tip}</body>
</html>
'''


def page(title: str, desc: str, body: str, on: str = '') -> str:
    links = [('index.html', 'Course')] + [(f'{m["slug"]}.html', f'M{m["n"]}') for m in MODULES] + [('glossary.html', 'Glossary'), ('reel.html', 'Reel')]
    nav = ''.join(f'<a href="{h}"{" class=on" if h == on else ""}>{t}</a>' for h, t in links)
    nav += f'<a class="film" href="{VIDEO}" target="_blank" rel="noopener">Film ↗</a>'
    return HEAD.format(title=html.escape(title), desc=html.escape(desc), nav=nav) + body + FOOT.format(video=VIDEO, tip=TIP_JS)


def timeline() -> str:
    segs = []
    for a, b, label, mod, chorus in SONG:
        left, width = a / LENGTH * 100, (b - a) / LENGTH * 100
        segs.append(f'<a href="m{mod}.html" class="{"chorus" if chorus else ""}" style="left:{left:.3f}%;width:{width:.3f}%" '
                    f'title="{label} · {tc(a)}–{tc(b)} · Module {mod}"><span class="n">{mod}</span>{label if width > 7 else ""}</a>')
    ticks = ''.join(f'<span>{tc(t)}</span>' for t in (0, 60, 120, 180, 240))
    return f'<div class="timeline" role="img" aria-label="The song, colored by module">{"".join(segs)}</div><div class="ticks">{ticks}</div>'


def module_page(i: int) -> str:
    m = MODULES[i]
    watch = ' · '.join(f'<a href="{VIDEO}?t={int(a)}" target="_blank" rel="noopener">{tc(a)}–{tc(b)} ↗</a>' for a, b in spans(m['n']))
    out = [f'<section class="mhead"><div class="label"><span class="now">Module {m["n"]}</span> · {html.escape(m["section"])} · '
           f'watch {watch}</div>'
           f'<h1>{html.escape(m["title"])}</h1><p class="prose">{gloss(m["intro"])}</p>{still(m["hero"], big=True)}</section>',
           f'<div class="lab" style="margin-top:28px"><b>LAB</b> · open <a href="{NB}/{m["file"]}"><code>notebooks/{m["file"]}</code></a> '
           f'in Jupyter. Each section below names the notebook section that runs it. '
           f'<span class="term-hint">Dotted words</span> show a definition when you hover or tap them.</div>']
    for c in m['concepts']:
        math = (f'<details class="math"><summary>The math</summary><div class="eq">$${c["tex"]}$$</div></details>'
                if c.get('tex') else '')
        keys = ['one', 'bridge', 'body', 'example', 'era', 'biz', 'lore']
        raw = [c['bridge'][1] if k == 'bridge' and c.get('bridge') else (c.get(k) or '') for k in keys]
        g = dict(zip(keys, gloss_in_order(raw)))
        bridge = (f'<div class="bridge"><div class="label">A familiar idea first</div>'
                  f'<h4>{html.escape(c["bridge"][0])}</h4><p>{g["bridge"]}</p></div>') if c.get('bridge') else ''
        example = f'<div class="example"><p>{g["example"]}</p></div>' if c.get('example') else ''
        era = f'<div class="aside era"><div class="label">Back then</div><p>{g["era"]}</p></div>' if c.get('era') else ''
        biz = f'<div class="aside biz"><div class="label">The business side</div><p>{g["biz"]}</p></div>' if c.get('biz') else ''
        lore = f'<div class="aside lore"><div class="label">Lore</div><p>{g["lore"]}</p></div>' if c.get('lore') else ''
        out.append(f'<section class="concept"><div class="label">{html.escape(c["k"])}</div>'
                   f'<div class="text"><h3>{html.escape(c["title"])}</h3><p class="one">{g["one"]}</p>'
                   f'{bridge}{g["body"]}{example}{math}'
                   f'<div class="lab"><b>LAB</b> · notebook {c["lab"]}</div></div>'
                   f'<div class="side">{still(c["t"])}{era}{biz}{lore}</div></section>')
    qs = ''.join(f'<li>{html.escape(q)}<details><summary>Answer</summary><p>{html.escape(a)}</p></details></li>' for q, a in m['checkpoint'])
    out.append(f'<section class="checkpoint"><h2>Checkpoint</h2><ol>{qs}</ol></section>')
    prev = f'<a href="{MODULES[i - 1]["slug"]}.html">← M{i - 1} · {html.escape(MODULES[i - 1]["title"])}</a>' if i else '<a href="index.html">← Course</a>'
    nxt = f'<a href="{MODULES[i + 1]["slug"]}.html">M{i + 1} · {html.escape(MODULES[i + 1]["title"])} →</a>' if i + 1 < len(MODULES) else '<a href="reel.html">The reel →</a>'
    out.append(f'<nav class="pager">{prev}{nxt}</nav>')
    return page(f'Baseline · M{m["n"]} {m["title"]}', m['card'], '\n'.join(out), f'{m["slug"]}.html')


def index_page() -> str:
    cards = ''.join(
        f'<a class="card" href="{m["slug"]}.html"><img src="stills/t{m["hero"]:06.2f}.jpg" alt="" loading="lazy" width="1600" height="900">'
        f'<div class="body"><div class="label"><span class="now">M{m["n"]}</span> · {html.escape(m["section"])}</div>'
        f'<h3>{html.escape(m["title"])}</h3><p>{html.escape(m["card"])}</p></div></a>' for m in MODULES)
    body = f'''
<section class="hero"><img src="stills/t005.50.jpg" alt="The film's first shot: a pen plotter draws y = 0, the first baseline" width="1600" height="900">
<div class="over"><div class="label"><span class="now">A first ML course</span> · for software developers · self-paced</div>
<h1>You were state of the art. <span class="mag">Now you're a baseline.</span></h1>
<p><em>Baseline</em> is a 4-minute music video that walks through more than thirty machine-learning techniques in order, from the 1958 perceptron to attention and its linear-time successors, each one replaced by the next. This course teaches those ideas, anchored to the film: every concept comes with a still from the moment the film shows it, and a Jupyter lab where you build it.</p>
<div class="cta"><a class="btn primary" href="{VIDEO}" target="_blank" rel="noopener">▶ Watch the film · 4:07</a><a class="btn" href="m0.html">Start with Module 0 →</a></div></div></section>

<section class="film" id="film">
<div class="label"><span class="now">The film</span> · <em>Baseline</em> · 4:07 · <a href="{VIDEO}" target="_blank" rel="noopener">open on YouTube ↗</a></div>
<div class="player"><iframe src="https://www.youtube-nocookie.com/embed/{VIDEO_ID}?rel=0" title="Baseline, the music video" loading="lazy"
  allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div>
<p class="prose">Watch it once all the way through before you start. You won't follow most of it yet, and that's the point: by the end of the course every shot will make sense. After that, each module links to the stretches of the film it covers, and every still's timecode jumps to that exact moment.</p>
</section>

<div class="prose">
<h2>How it works</h2>
<p><strong>Watch</strong> the parts of the film a module covers (each module and every still link to their timestamps). <strong>Read</strong> the module page: every concept is anchored to a still from the film, so you always know where you are. <strong>Run</strong> the lab notebook: you build each technique yourself, at a size that fits your machine, and see the same kind of real numbers the film puts on screen.</p>
<p>No ML background is assumed. You need to read Python and be comfortable at a terminal. The math is shown, but each idea is also explained in code, and every lab has small exercises with automatic checks: write a function, run the cell, see ✓ or what went wrong.</p>
</div>

<h2>The song, reordered for learning</h2>
<p class="prose">The film goes in historical order. The course goes in the order ideas depend on each other, so a module can cover several stretches of the film. The strip below is the song; each stretch is colored by the module that teaches it.</p>
{timeline()}
<div class="cards">{cards}</div>

<div class="prose">
<h2 id="setup">Setup</h2>
<p>You need Python 3.11+ and <a href="https://docs.astral.sh/uv/">uv</a>. From the repository root:</p>
<pre><code>cd course
uv sync                         # torch, transformers, scikit-learn, jupyterlab …
uv run jupyter lab notebooks/   # open 00_the_loop.ipynb first</code></pre>
<p>Each notebook detects your hardware and picks a <strong>tier</strong>. Every lab reads its model sizes from that tier, so the same notebook runs everywhere:</p>
<div class="scroll"><table>
<tr><th>Tier</th><th>Detected when</th><th>What you get</th><th>Downloads</th></tr>
<tr><td><code>laptop</code></td><td>anything else (CPU, or a base Apple-silicon GPU)</td><td>small nets, data subsets, Qwen2.5-0.5B. Each lab runs in minutes; tested on CPU.</td><td>≈ 2.5 GB</td></tr>
<tr><td><code>mac-max</code></td><td>Apple MPS with ≥ 32 GB unified memory</td><td>wider and deeper nets, full CIFAR-10 with residual blocks, Qwen2.5-1.5B, 16K-token haystacks</td><td>≈ 5 GB</td></tr>
<tr><td><code>cuda</code></td><td>NVIDIA GPU with ≥ 12 GB VRAM</td><td>biggest configs, Qwen2.5-7B in bf16, 32K-token haystacks</td><td>≈ 17 GB</td></tr>
</table></div>
<p>To force a tier or device, set it before launching: <code>BASELINE_TIER=laptop BASELINE_DEVICE=cpu uv run jupyter lab notebooks/</code>. The sizes live in one table, <code>TIERS</code> in <code>notebooks/blcourse.py</code>, so change it if you want to experiment.</p>

<h2>Self-paced plan</h2>
<div class="scroll"><table>
<tr><th>Module</th><th>Read</th><th>Lab (laptop tier, CPU)</th><th>You will have built</th></tr>
<tr><td>M0 · The Loop</td><td>40 min</td><td>&lt; 1 min</td><td>hand-written rules vs a learned template; gradient descent; a digit classifier with softmax and cross-entropy; mini-batch training</td></tr>
<tr><td>M1 · One Line, Then Many</td><td>45 min</td><td>≈ 3 min</td><td>backprop in numpy; XOR wired by hand; HOG + SVM vs a CNN on CIFAR-10; a residual net</td></tr>
<tr><td>M2 · Sequences</td><td>60 min</td><td>≈ 4 min</td><td>n-grams, an LSTM, seq2seq ± attention, a tiny GPT, one leaderboard; BERT's mask; RoPE</td></tr>
<tr><td>M3 · Generating Data</td><td>20 min</td><td>&lt; 1 min</td><td>a GAN that collapses and a diffusion model that doesn't</td></tr>
<tr><td>M4 · A Real LLM</td><td>60 min</td><td>≈ 5 min + LLM download</td><td>INT4 quantization, LoRA, few-shot prompting and DPO on a real LLM</td></tr>
<tr><td>M5 · Scale and Long Context</td><td>45 min</td><td>≈ 2 min</td><td>your own scaling law, FlashAttention from scratch, a needle-in-a-haystack map, linear attention</td></tr>
</table></div>
<p>Every module ends with checkpoint questions with folded answers, and every lab has <em>Predict first</em> prompts and <em>Your turn</em> exercises with automatic checks. The reference solutions are in <code>notebooks/exercises.py</code>. The <a href="reel.html">reel</a> lists every still in film order, with the module that explains it.</p>
</div>
'''
    return page('Baseline · ML for Developers', 'A first machine-learning course for software developers, taught through the Baseline music video.', body, 'index.html')


def reel_page() -> str:
    items = []
    for k in sorted(CAPS, key=float):
        t, c = float(k), CAPS[k]
        n = module_at(t)
        items.append(f'<div><div class="label"><a class="mod" href="m{n}.html">Module {n}</a></div>{still(t)}</div>')
    body = (f'<div class="prose" style="margin-top:48px"><div class="label"><span class="now">The reel</span> · {len(CAPS)} stills · film order</div>'
            f'<h1>Every frame <span class="mag">we teach from</span></h1><p>Click a timecode to watch that moment, or a module to read about it.</p></div>'
            f'<div class="reel" style="margin-top:36px">{"".join(items)}</div>')
    return page('Baseline · The Reel', 'Every still from the Baseline music video used in the course, in film order.', body, 'reel.html')


def glossary_page() -> str:
    used: dict[str, set[int]] = {}
    for m in MODULES:
        for c in m['concepts']:
            for f in ('one', 'body', 'example', 'era', 'biz', 'lore'):
                for key, _ in TERM.findall(c.get(f) or ''):
                    used.setdefault(key.strip().lower(), set()).add(m['n'])
            if c.get('bridge'):
                for key, _ in TERM.findall(c['bridge'][1]):
                    used.setdefault(key.strip().lower(), set()).add(m['n'])
    rows = []
    for key in sorted(GLOSSARY, key=str.lower):
        slug = re.sub(r'[^a-z0-9]+', '-', key.lower()).strip('-')
        mods = ' '.join(f'<a href="m{n}.html">M{n}</a>' for n in sorted(used.get(key.lower(), ())))
        rows.append(f'<div class="entry" id="{slug}"><dt>{html.escape(key)}</dt><dd>{html.escape(GLOSSARY[key])}'
                    f'{f" <span class=mods>{mods}</span>" if mods else ""}</dd></div>')
    body = (f'<div class="prose" style="margin-top:48px"><div class="label"><span class="now">Glossary</span> · {len(GLOSSARY)} terms</div>'
            f'<h1>Words that <span class="mag">might not have stuck</span></h1>'
            f'<p>Every dotted word in the module pages is defined here. The tags show which modules use it.</p>'
            f'<dl class="glossary">{"".join(rows)}</dl></div>')
    return page('Baseline · Glossary', 'Plain-language definitions of every term in the Baseline ML course.', body, 'glossary.html')


if __name__ == '__main__':
    (SITE / 'index.html').write_text(index_page())
    for i, m in enumerate(MODULES):
        (SITE / f'{m["slug"]}.html').write_text(module_page(i))
    (SITE / 'reel.html').write_text(reel_page())
    (SITE / 'glossary.html').write_text(glossary_page())
    print('wrote', ', '.join(['index.html'] + [f'{m["slug"]}.html' for m in MODULES] + ['reel.html', 'glossary.html']))
