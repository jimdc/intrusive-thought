# Intrusive Thought

**Turn a concept up until a language model can’t think of anything else — or down, until it won’t say
it.** A real **Gemma 2 (2B)**, steered live-on-a-slider, two different ways. Ten knobs total, each one
teaching a real lesson about how a language model represents meaning, or about the mechanics of
steering it.

🔗 **Live:** https://jimdc.github.io/intrusive-thought/

The page is fully static — every completion is precomputed into `data/steers.json` and
`data/steer-actadd.json`, so there is no server, no API call, and nothing to install to *use* it. Open
`index.html` and drag.

---

## What you’re looking at

Every dial on this page does the exact same thing to the model: a forward hook adds a fixed direction
to one layer’s residual stream while it generates, scaled by the slider. Let go at 0 and the hook adds
nothing — the model is exactly, provably itself again, because nothing was ever written to a weight.
That mechanism is identical across all ten knobs. **What differs is only how each direction was found**,
and the two ways of finding one teach two different lessons.

- **SAE feature** (seven chips) — the direction is one of ~16,000 features a sparse autoencoder learned
  by decomposing the model’s own activations. Pick one, and you’re steering a direction the model already
  uses on its own terms. These seven teach the *structure* of stored meaning: force/forbid asymmetry,
  feature-splitting, meaning that lives above spelling.
- **ActAdd contrast pair** (three chips) — no autoencoder, no training. The direction is just the
  difference between the model’s activations on two short prompts you write yourself — e.g. `"Love.
  Kindness. Warmth."` minus `"Hate. Cruelty. Coldness."` This is [activation addition](https://arxiv.org/abs/2308.10248)
  (Turner et al., 2023), and it teaches *reversibility*: the purest, weight-untouching rung of the
  model-control stack, with no dictionary of features to search.

### The SAE-feature bank

A sparse autoencoder decomposes the model’s internal state into ~16,000 **features** — directions the
model itself uses to represent concepts. Pick one with a chip, then drag its slider:

- **right (force):** add that feature’s direction to the residual stream, harder and harder. The model
  fixates, then floods, then — past a point — collapses into a stutter.
- **left (forbid):** subtract it. The model can usually still reach the meaning by another route, so it
  paraphrases *around* the blocked word instead of going silent.

Each of the seven chips is chosen to make one phenomenon vivid:

| chip | the lesson | grounded in |
|---|---|---|
| **Golden Gate Bridge** | one feature, turned up, can hijack the whole personality | [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) |
| **drugs** | a concept *splits* into finer ones (cocaine, heroin, meth) when amplified | [Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features/) |
| **death** | forbid a thought and it *euphemizes* — the meaning is stored many ways | [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) |
| **Spanish** | one knob fires in *any language* — meaning lives above spelling | [Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/) |
| **scripture** | force it and the *voice changes genre* — ordinary chatter starts to preach | [Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/) |
| **gratitude** | even a *mood* is a real, movable direction (“fortunately… luckily…”) | [Park et al. 2024](https://arxiv.org/abs/2311.03658) |
| **flattery** | some knobs are *behaviors*, not topics — turn up sycophancy and it gushes | [Sharma et al. 2023](https://arxiv.org/abs/2310.13548) |

**The asymmetry is the real lesson.** Forcing and forbidding are *different operations*: forcing has no
ceiling (so it eventually drowns the computation and degenerates), while forbidding only blocks one route
to a redundantly-stored meaning (so the model reroutes to a euphemism). They are not mirror images.

### The ActAdd contrast-pair bank

Three chips, each a pair of opposite prompts (Love/Hate, Formal/Casual, Playful/Serious). The steering
vector is just `activation(positive prompt) − activation(negative prompt)`, mean-pooled at one middle
layer — no sparse autoencoder, no training run, no 16,000-entry dictionary to search. Drag the slider and
the model leans toward whichever prompt the sign points at; the page shows you the actual two prompts
whenever you pick a chip here, because *that pair is the entire method* — there’s no hidden feature index
behind it, only two sentences someone wrote.

**The reversibility is the real lesson.** At coefficient 0 the forward hook is a no-op — not an
approximation of the baseline, the baseline exactly. That’s the whole promise of activation addition: a
control lever that leaves no residue, because it was never written to the weights it perturbs.

---

## How it works

- **Model:** [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) (a *base* model, gated on
  Hugging Face) for both banks.
- **SAE feature bank:** [Gemma Scope](https://deepmind.google/models/gemma/gemma-scope/) JumpReLU SAEs,
  residual stream, **layer 20**, width **16k** (`gemma-scope-2b-pt-res-canonical`,
  `layer_20/width_16k/canonical`). A forward hook on layer 20 adds `strength · W_dec[feature]` to the
  residual stream during generation. Decoder rows are unit-norm and the residual norm at L20 is ≈ 376,
  so a strength around ~190 makes the model fixate and ~420 breaks it. Feature labels/indices were
  located with help from [Neuronpedia](https://www.neuronpedia.org/).
- **ActAdd contrast-pair bank:** [Turner et al., *Activation Addition* (2023), arXiv:2308.10248](https://arxiv.org/abs/2308.10248).
  Each concept’s direction is the difference of the model’s mean-pooled residual-stream activations on a
  contrast pair, read at **layer 13** (the midpoint of Gemma 2 2B’s 26 layers). A forward hook adds
  `coefficient · direction` to that layer during generation; coefficient 0 is the identity.
- **Precompute:** for each concept and a fixed ladder of strengths/coefficients, the completion is
  generated once and saved to `data/steers.json` (SAE) or `data/steer-actadd.json` (ActAdd). The browser
  just replays them — no model or server runs at view time.

### Honest limits

- A single width-16k feature is a *shard* of a concept, not the whole of it; feature labels are
  auto-generated and approximate. **Trust the effect, not the label.**
- A clean Neuronpedia label does **not** guarantee a feature actually steers output — every concept here
  was empirically steer-tested, and several plausible candidates were dropped because they didn’t move.
- SAE steering is *illustrative*, not state-of-the-art control: it has off-target effects and a narrow
  “sweet spot” of strengths (see [Anthropic, *Evaluating Feature Steering*](https://www.anthropic.com/research/evaluating-feature-steering)).
- The SAE examples are deliberately pointed (drugs, death, flattery) — specific, token-distinctive
  concepts steer far more cleanly than abstract ones, and the contrast is part of the point.
- ActAdd directions are cheap to build but coarser than a learned SAE feature: a mean-pooled contrast
  pair captures whatever the two prompts differ on, not necessarily only the intended concept — and it
  isn’t unit-norm the way an SAE decoder row is. At layer 13 the direction’s own norm (~47) is already
  a sizeable fraction of the residual stream’s (~293), so the coefficient ladder here (±4, ±2) was tuned
  down from an initial ±12/±6 that collapsed almost every completion into repetition. Even at ±4 the
  “sweet spot” is narrower and less even than the SAE bank’s: some sides land a clean, legible lean, and
  some collapse before they get there — that asymmetry is itself real, not a bug to paper over.

---

## Regenerating the data (optional)

You only need this to change the concepts, strengths, or coefficients; the shipped JSON is enough to run
the site.

**SAE features** — `tools/build_bidir.py`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers sae_lens huggingface_hub

# gemma-2-2b is gated: accept the license on its HF page, then provide a token.
# Put HF_TOKEN=hf_xxx in a .secrets.env file (gitignored) or export it.
export HF_TOKEN=hf_...

python tools/build_bidir.py                 # rebuild all concepts
python tools/build_bidir.py spanish death   # rebuild only some, merge into the rest
```

Each concept (feature index, topic-forcing prompt, lesson, paper) is defined at the top of
`tools/build_bidir.py`.

**ActAdd contrast pairs** — `build_actadd.py`. It only needs `torch` + `transformers` (no SAE library,
no gated download beyond the base model), and reads `google/gemma-2-2b` from the local Hugging Face
cache offline:

```bash
python3 -m venv .venv-actadd && source .venv-actadd/bin/activate
pip install torch transformers

python build_actadd.py --selftest             # vector-math check, no model, no download
HF_HUB_OFFLINE=1 python build_actadd.py        # full run — writes data/steer-actadd.json
```

Each concept (contrast pair, pole labels, prompt, lesson) is defined at the top of `build_actadd.py`.

## Run locally

```bash
python3 -m http.server 8000   # then open http://localhost:8000/
```

## Credits

[Gemma Scope](https://deepmind.google/models/gemma/gemma-scope/) (Google DeepMind) ·
[Neuronpedia](https://www.neuronpedia.org/) ·
[Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) (Anthropic) ·
[Turner et al., *Activation Addition* (2023)](https://arxiv.org/abs/2308.10248). Papers are linked in the
tables and prose above.

**History:** ActAdd steering here used to be a separate demo, `steering-wheel`. It’s now folded into this
page as the second bank, sharing the same slider and the same model — the point of merging them was that
the *comparison* between the two ways of finding a direction is worth more than two look-alike demos.

## License

MIT — see [`LICENSE`](LICENSE).
