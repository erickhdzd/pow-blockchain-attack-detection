# PoW Blockchain Attack Detection

Version: 2.0.0

## Overview

Research software supporting the paper
"Early Detection of Proof-of-Work Blockchain Attacks Using Temporal Deep Learning"
(ACDSA 2027).

## Validated environment

- Debian 13
- Python 3.15.0
- PyTorch 2.14.1+cu130
- CUDA 13.0
- NVIDIA GeForce RTX 4060 Ti

## Installation

Create the virtual environment:

    python3.15 -m venv .venv

Install dependencies:

    .venv/bin/python -m pip install -r requirements.txt

## Project structure

- configs/: simulation and training configurations
- src/simulator/: blockchain and attack simulation
- src/dataset/: dataset generation and partitioning
- src/training/: machine learning experiments
- src/analysis/: statistical analysis and paper artifacts
- tests/: automated tests

## Verification

    .venv/bin/python -m ruff check src tests
    .venv/bin/python -m ruff format --check src tests
    .venv/bin/python -m mypy src tests
    .venv/bin/python -m pytest -q
    .venv/bin/python -m pip check

## Validation results

- 179 tests passed
- 1 test skipped
- Ruff: clean
- Mypy: clean
- pip check: clean

## Reproducibility

The scientific results associated with the submitted ACDSA 2027
paper are preserved. This release updates the software environment
without claiming new experimental results.

## Release v2.0.0

- Python 3.15.0 compatibility
- PyTorch 2.14.1+cu130
- CUDA 13.0 support
- Consolidated .venv environment
- Updated dependency versions
- Static analysis and automated test validation
