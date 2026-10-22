# ML Assignment 2 - Reproduction Instructions

## Environment

- Python 3.13.4
- Package versions: `library_versions.txt`
- Data split: `results/split.npz`
- Saved experiment outputs and figures: `results/`

## Setup

Run from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy matplotlib scikit-learn
```

## Reproduction commands

Run commands from the repository root.

The following commands use the saved data split and generate the baseline model outputs:

```powershell
python .\src\tree_baseline.py
python .\src\bagging.py
python .\src\boosting.py
```

Generate the baseline comparison figures:

```powershell
python .\src\make_figures.py
```

Additional scripts for bias-variance analysis, SAMME, error analysis, and figures are available in `src/`. Consult each script before running it to identify required input files and avoid overwriting saved outputs or checkpoints.

## Saved artifacts

- `results/split.npz` - saved train, validation, and test split
- `results/` - saved metrics, experiment outputs, checkpoints, CSV files, and figures
- `src/` - experiment and plotting scripts
- `library_versions.txt` - recorded package versions
- `logs/reproduction_log.txt` - environment and saved-result inventory

## Log scope

The committed log records the environment and saved-result inventory. It is not a record of a complete clean rerun of every experiment.
