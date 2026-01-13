# Contributing to RLM Self-Distillation

Thank you for your interest in contributing to this research project!

## Getting Started

1. **Fork the repository** and clone your fork locally

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Install Ollama** and pull required models:
   ```bash
   ollama pull llama3.2:3b
   ```

4. **Run tests** to ensure everything works:
   ```bash
   PYTHONPATH=. uv run pytest tests/
   ```

## Running Experiments

```bash
# List available experiments
python main.py --list

# Run an experiment
python main.py --experiment exp004 --cola 50 --pii 30

# View results in MLflow
mlflow ui
```

## Project Structure

```
self_distill/          # Core library code
├── datasets/          # Dataset loaders (CoLA, PII, GSM8K)
├── clients/           # LLM client wrappers
└── tracking/          # MLflow experiment tracking

experiments/           # Experiment scripts
tests/                 # Test suite (mirrors self_distill structure)
```

## Adding New Experiments

1. Create a new file in `experiments/` following the naming convention: `exp00X_description.py`
2. Add the experiment to `EXPERIMENTS` dict in `main.py`
3. Include MLflow tracking for metrics
4. Generate a PDF report with results

## Adding New Datasets

1. Create a new file in `self_distill/datasets/`
2. Extend `BaseDataset` class from `base.py`
3. Add to the `DATA` enum in `__init__.py`
4. Add corresponding tests in `tests/datasets/`

## Code Style

- Use type hints for function signatures
- Add docstrings to public functions and classes
- Follow PEP 8 conventions
- Run tests before submitting PRs

## Test Data

The PII dataset uses synthetic test data:
- SSNs use patterns like `123-45-6789` (not real)
- Phone numbers use `555-XXX-XXXX` (reserved test prefix)
- Emails use `@example.com` or `@test.com` domains

**Never include real personal information in test data.**

## Reporting Issues

When reporting bugs, please include:
- Python version (`python --version`)
- Ollama version (`ollama --version`)
- Steps to reproduce the issue
- Full error traceback

## Questions?

Open an issue for questions about the codebase or research methodology.
