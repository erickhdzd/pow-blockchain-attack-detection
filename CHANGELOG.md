# Changelog

## [2.0.0] - 2026-10-14

### Changed
- Updated the validated environment to Python 3.15.0.
- Updated PyTorch to 2.14.1+cu130 (CUDA 13.0).
- Updated and pinned Python dependencies.
- Applied Python 3.15 compatibility and typing corrections.
- Updated simulation, dataset, training, analysis, and test code.
- Consolidated the active virtual environment as `.venv`.
- Removed PyCharm project metadata from Git tracking.
- Added installation and verification documentation.

### Validation
- Ruff: passed.
- Ruff formatting: passed.
- Mypy: passed.
- Pytest: 179 passed, 1 skipped.
- pip check: no broken requirements.

### Scientific reproducibility
- Previously generated scientific artifacts are retained.
- No new experimental results are claimed.
- Numerical equivalence with the previous environment has not
  been independently established.
