# WQ Tool

`WQ Tool` is a production-style MVP for automated alpha research on local OHLCV data. It provides:

- schema-normalized CSV ingestion
- auxiliary group/factor/mask metadata alignment
- reusable vectorized feature operators
- a safe alpha expression parser and evaluator
- hybrid alpha generation
- a BRAIN-compatible local simulation pipeline with D0/D1/Fast D1 profiles
- neutralization, submission-style tests, evaluation, filtering, and deduplication
- simulation signature caching to avoid duplicate evaluation work
- SQLite persistence
- a resumable CLI pipeline

## Quick Start

```bash
python -m pip install -e .[dev]
python main.py run-full-pipeline --config config/dev.yaml
python main.py top --config config/dev.yaml --limit 10
python main.py report --config config/dev.yaml --limit 5
```

Sample data, expressions, and a walkthrough are included under `examples/`.
