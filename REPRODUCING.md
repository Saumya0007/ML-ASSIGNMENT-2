# ML Assignment 2 — Reproduction Instructions

## Environment
- Python 3.13.4
- Package versions are recorded in `library_versions.txt`.
- Saved experiment outputs and the data split are in `results/`.

## Setup

Run from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy matplotlib scikit-learn
```

## Reproduction

The experiment scripts are located in `src/`. Run the required scripts from the repository root in the order needed by their dependencies. For example, inspect each script before rerunning it because some scripts regenerate saved results or require existing checkpoints.

```powershell
python .\src\data_setup.py
python .\src\bagging.py
python .\src\boosting.py
```

Additional analysis and figure-generation scripts are also available in `src/`. The saved split, experiment outputs, and checkpoints are stored in `results/`.

## Log

The environment and saved-result inventory are recorded in `logs/reproduction_log.txt`.
