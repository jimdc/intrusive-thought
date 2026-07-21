# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.
- Site is fully static (root `index.html`, `static/`, `data/`) and deploys via `.github/workflows/pages.yml` (GitHub Actions Pages deploy), not the legacy Jekyll builder — `.nojekyll` at root reinforces this. The repo's Pages source setting (Settings → Pages) must be "GitHub Actions" for the workflow to take effect.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
