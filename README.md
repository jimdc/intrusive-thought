# Intrusive Thought

**Turn a concept up until a language model can’t think of anything else — or down, until it won’t say
it.** A real **Gemma 2 (2B)**, steered live-on-a-slider by its own internal features. Seven knobs, each
one teaching a different, real lesson about how a language model represents meaning.

🔗 **Live:** https://jimdc.github.io/intrusive-thought/

The page is fully static — every completion is precomputed into `data/steers.json`, so there is no
server, no API call, and nothing to install to *use* it. Open `index.html` and drag.

---

## What you’re looking at

A sparse autoencoder (SAE) decomposes the model’s internal state into ~16,000 **features** — directions
the model itself uses to represent concepts. Pick one with a chip, then drag its slider:

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

---

## How it works

- **Model:** [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) (a *base* model, gated on
  Hugging Face).
- **Features:** [Gemma Scope](https://deepmind.google/models/gemma/gemma-scope/) JumpReLU SAEs, residual
  stream, **layer 20**, width **16k** (`gemma-scope-2b-pt-res-canonical`, `layer_20/width_16k/canonical`).
- **Steering:** a forward hook on layer 20 adds `strength · W_dec[feature]` to the residual stream during
  generation. Decoder rows are unit-norm and the residual norm at L20 is ≈ 376, so a strength around
  ~190 makes the model fixate and ~420 breaks it.
- **Precompute:** for each concept and a fixed ladder of strengths, the completion is generated once and
  saved to `data/steers.json`. The browser just replays them. Feature labels/indices were located with
  help from [Neuronpedia](https://www.neuronpedia.org/).

### Honest limits

- A single width-16k feature is a *shard* of a concept, not the whole of it; feature labels are
  auto-generated and approximate. **Trust the effect, not the label.**
- A clean Neuronpedia label does **not** guarantee a feature actually steers output — every concept here
  was empirically steer-tested, and several plausible candidates were dropped because they didn’t move.
- SAE steering is *illustrative*, not state-of-the-art control: it has off-target effects and a narrow
  “sweet spot” of strengths (see [Anthropic, *Evaluating Feature Steering*](https://www.anthropic.com/research/evaluating-feature-steering)).
- The examples are deliberately pointed (drugs, death, flattery) — specific, token-distinctive concepts
  steer far more cleanly than abstract ones, and the contrast is part of the point.

---

## Regenerating the data (optional)

You only need this to change the concepts or strengths; the shipped `data/steers.json` is enough to run
the site.

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

## Run locally

```bash
python3 -m http.server 8000   # then open http://localhost:8000/
```

## Credits

[Gemma Scope](https://deepmind.google/models/gemma/gemma-scope/) (Google DeepMind) ·
[Neuronpedia](https://www.neuronpedia.org/) ·
[Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) (Anthropic). Papers are linked
in the table above.

## License

MIT — see [`LICENSE`](LICENSE).
