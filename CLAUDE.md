# intrusive-thought — dev notes

## Shared Gemma Scope loader (adopt when development resumes)
The Gemma-2-2b + Gemma Scope SAE loading and layer-20 hook in
`tools/{build_bidir,curate4,probe_slot}.py` is duplicated boilerplate — the same ~6 lines live in
sibling projects too. A single shared extractor now exists, **`gemma_scope_local`**, exposing:

- `load()` → `Extractor`
- `Extractor.features(text, topk)` → `[(feature_index, max_activation), …]`
- `Extractor.steer(text, feature_index, alpha)` → steered completion (the force/forbid used here)

Import it via the **`GEMMA_SCOPE_LOCAL`** env var (point it at the extractor's directory); the working
client pattern is in the public `semantic-axes` repo at `tools/build_sae_local.py`. When you next edit
these tools, replace the inline model+SAE setup with that import instead of re-pasting it. It reads the
global Hugging Face cache, so there's no per-project re-download of the 9.8 GB weights.

(The extractor is kept outside any public repo on purpose — reach it only through the env var, never a
hardcoded path, so nothing about the local setup leaks into this public repo.)
