# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Site is fully static (root `index.html`, `static/`, `data/`) and deploys via `.github/workflows/pages.yml` (GitHub Actions Pages deploy), not the legacy Jekyll builder — `.nojekyll` at root reinforces this. The repo's Pages source setting (Settings → Pages) must be "GitHub Actions" for the workflow to take effect.
- **Two data banks, one client-side render path.** The site loads two precomputed JSON files —
  `data/steers.json` (SAE features, built by `tools/build_bidir.py`) and `data/steer-actadd.json`
  (ActAdd contrast pairs, built by `build_actadd.py`) — and `static/app.js` normalizes both into a
  shared "tour" shape (`normalizeSae`/`normalizeActadd`) before one `render()` drives either kind of
  knob. Keep both generator scripts' own JSON shapes stable; do the normalization in `app.js`, not by
  reshaping the generated data.
- **No model or server at view time.** Every completion is precomputed offline and replayed from
  JSON; `index.html` is a static page. See each build script's own header comment for its precompute
  command and required env (`HF_TOKEN` for the gated SAE path, `HF_HUB_OFFLINE=1` for the ungated
  ActAdd path).
- **`build_actadd.py` picks the fastest available device** (CUDA, then MPS, then CPU) and loads with
  `attn_implementation="sdpa"` — eager attention on MPS was the actual bottleneck (~40+ min for one
  generation, not resource contention) until SDPA cut a 40-token generation to ~15-45s. On CPU with
  contention the 3-concept x 5-coefficient ladder can still take an hour+; check `headroom.py` first.
  `taskpolicy -b` throttles Metal/GPU dispatch far more than CPU work — for this specific job (short,
  bounded, GPU-bound) it turned a ~10 min run into an hour+; prefer running it unthrottled once
  headroom is clear rather than reaching for `taskpolicy -b` by default.
- **The ActAdd contrast vector isn't unit-norm.** Unlike an SAE decoder row, its magnitude depends on
  the contrast pair's own text, so a coefficient that works for one concept can be far past the
  "sweet spot" for another. `COEFFICIENT_LADDER` was tuned down from ±12/±6 (near-total collapse) to
  ±4/±2 by comparing the direction's norm against the residual stream's own norm at the target layer
  — see the comment above `COEFFICIENT_LADDER` in `build_actadd.py` for the numbers. If you change the
  contrast pairs, sanity-check the new direction's norm the same way before trusting the ladder.
- **e2e:** `npm install && npx playwright install chromium && npx playwright test` — `tests/e2e/`
  drives the page at both 390px and 1440px, checks for horizontal overflow, and writes screenshots to
  `tests/e2e/screenshots/` (committed, for PR review). `playwright.config.js` serves the repo with
  `python3 -m http.server`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
