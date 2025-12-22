<!-- Copied/created to help AI coding agents be productive in this small notebook-driven repo -->
# Copilot / AI Agent Instructions — Multimodal ML Pipeline

Purpose
- Help an AI agent make safe, small, observable changes to this repository's preprocessing pipeline.

Big picture (what you'll find)
- This is a small, notebook-first preprocessing project. The main artifact is `Preprocessing_Pipeline.ipynb` which orchestrates data cleaning and image handling.
- Raw data lives under `Data/`:
  - `Data/socal2.csv` — tabular metadata used by preprocessing steps.
  - `Data/socal2/socal_pics/` — image assets referenced by the CSV.

Key workflows and developer commands
- Activate the project virtualenv (Windows):

  ```powershell
  .venv\Scripts\Activate
  python -m pip install -r requirements.txt  # if requirements.txt is added
  ```

- Primary workflow: open and run `Preprocessing_Pipeline.ipynb` in VS Code / Jupyter. Make iterative edits in small cells, run locally, and commit notebook diffs.
- If you add Python modules or scripts, keep them under a new top-level folder (e.g., `src/` or `notebook_helpers/`) and update README/requirements accordingly.

Project-specific conventions and patterns
- Notebook-first: prefer small, testable cell changes over large refactors. When adding reusable code, extract it to a `.py` module and import from the notebook.
- Data paths are relative to the repo root. Use `Path('Data') / 'socal2.csv'` or similar to remain cross-platform.
- Avoid editing or moving raw data (`Data/`) unless the change is explicitly about preprocessing (e.g., corrected filenames). Large data changes must be explained in the commit message.

Integration points & external dependencies
- There is no declared `requirements.txt` yet; expect common data stack (pandas, numpy, pillow/opencv) if new code is added. Prefer adding a `requirements.txt` when introducing new packages.
- The repo contains a `.venv/` directory — use it for local runs but do not check in changes to the venv.

Safe editing rules for AI agents
- Keep changes small and incremental. For notebooks, separate logic changes into their own cell and run the notebook to confirm behavior.
- Do not add or modify large binary files or images under `Data/`.
- When changing `Preprocessing_Pipeline.ipynb`, keep outputs cleared for large diffs unless outputs are necessary for review.
- Always include a short summary in commits describing why the change was made and which notebook cells to run for verification.

Concrete examples (what to do / what not to do)
- Good: Extract a helper function used in multiple cells into `notebook_helpers/io.py`, add tests, import it in the notebook.
- Good: Add a new notebook cell that loads `Data/socal2.csv`, shows `df.head()`, and drops rows with missing image paths.
- Bad: Replace or delete the `Data/socal2/` folder or bulk-add many images without explanation.

Where to look for more context
- Start with `Preprocessing_Pipeline.ipynb` to understand intended preprocessing steps.
- Inspect `Data/socal2.csv` to learn column names and how images are referenced.

If you are unsure
- Ask a human reviewer before making large data or schema changes. When in doubt, propose a small PR with a focused change and instructions to reproduce the result.

Next steps for contributors
- If you add any non-trivial dependencies or scripts, update or add `requirements.txt` and a short `README.md` at the repository root documenting how to run the notebook and reproduce preprocessing.

---
Please review and tell me if you'd like more/less detail or examples specific to a particular preprocessing step.
