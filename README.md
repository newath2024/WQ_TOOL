# WQ Tool

`WQ Tool` là một alpha research framework theo phong cách WorldQuant/BRAIN, chạy local trên dữ liệu OHLCV và metadata phụ trợ. Mục tiêu của repo là cung cấp một codebase đủ sạch để nghiên cứu alpha lặp lại, đánh giá có traceability, và mở rộng dần sang universe lớn hơn, operator nhiều hơn, và workflow tối ưu hóa theo memory.

## Dự án này dành cho ai

- quantitative researcher cần pipeline generate -> evaluate -> rank -> mutate
- engineer muốn một khung local, minh bạch, dễ debug
- contributor cần một repo có service boundaries đủ rõ để tiếp tục mở rộng

## Tính năng chính

- nạp dữ liệu CSV local theo schema chuẩn hóa `timestamp,symbol,open,high,low,close,volume`
- nạp metadata phụ trợ cho group/factor/mask
- alpha DSL an toàn, không dùng raw `eval`
- template, grammar, mutation, và guided generation
- simulation local kiểu BRAIN với `d0`, `d1`, `fast_d1`, neutralization, submission-style tests
- ranking, dedup, stability checks, behavioral novelty, adaptive memory
- SQLite persistence cho runs, lineage, metrics, cache, diagnoses, pattern memory
- CLI đầy đủ cho load/generate/evaluate/report/memory/lineage/mutate/pipeline

## Kiến trúc mức cao

```text
main.py            Thin compatibility wrapper
cli/               Argparse wiring và output formatting
services/          Workflow orchestration và typed service results
alpha/             Parser / AST / evaluator
backtest/          Portfolio simulation và metrics
data/              Ingestion / validation / splitting
evaluation/        Filtering / critic / ranking / submission tests
features/          Operators và transforms
generator/         Template / grammar / mutation / guided generation
memory/            Pattern memory và adaptive scoring
storage/           SQLite schema, repository, history persistence
config/            Dev / research / strict profiles
docs/              Kiến trúc, pipeline, config, development notes
tests/             Unit + integration smoke tests
```

## Vòng đời pipeline

1. `load-data`: nạp và validate dataset + metadata
2. `generate`: sinh alpha candidates mới
3. `evaluate`: parse -> evaluate -> backtest -> filter -> dedup -> select
4. `report` / `top`: xem kết quả và lý do ranking
5. `memory-*` / `lineage`: xem pattern memory và ancestry
6. `mutate`: sinh thế hệ tiếp theo từ alpha tốt hơn
7. `run-full-pipeline`: chạy end-to-end

## Luồng alpha expression

1. Chuỗi expression được parse thành custom AST
2. Validator kiểm tra operator, field, depth, group-field usage
3. Evaluator áp AST lên field matrices theo symbol/date
4. Signal đi qua simulation pipeline
5. Metrics validation là driver chính cho filtering và adaptive memory
6. Test split chỉ để audit, không lái generator

## Workflow generation / evaluation / ranking

- generation có 4 nhánh: guided mutation, memory templates, random exploration, novelty search
- critic gắn fail tags như `high_turnover`, `weak_validation`, `low_stability`
- memory tách expression thành family/operator/field/lookback/wrapper/subexpression genes
- filtering áp hard filters, data sufficiency, submission-style robustness
- ranking ưu tiên:
  - validation fitness
  - submission pass count
  - validation sharpe
  - behavioral novelty
  - lower complexity

## Memory và mutation

- mỗi alpha sau evaluate được lưu history, diagnosis, rejection reasons, lineage, và pattern membership
- generator vòng sau giảm sampling của pattern fail lặp lại và tăng sampling cho building blocks hiệu quả
- mutation policy phản ứng theo critic hints, ví dụ:
  - turnover cao -> smooth/slow signals
  - overfit -> đơn giản hóa
  - correlation cao -> đổi feature/operator family

## Input data format

OHLCV canonical:

```csv
timestamp,symbol,open,high,low,close,volume
2021-01-01,AAA,100.0,101.0,99.5,100.5,120000
```

Metadata phụ trợ có thể là:

- group fields: `sector, industry, country, subindustry`
- factor fields: `beta, size, volatility, liquidity`
- mask fields: `core_mask, liquid_mask`

Sample dataset trong `examples/` chỉ là smoke fixture để chạy demo và test. Nó không đại diện cho universe nghiên cứu thật.

## Config profiles

- `config/dev.yaml`: demo/dev profile, thresholds lỏng hơn, sample universe nhỏ
- `config/research.yaml`: default research profile, thực tế hơn cho local screening
- `config/strict.yaml`: audit profile, filters chặt hơn
- `config/default.yaml`: mirror của `research.yaml` để giữ workflow mặc định

Loader hỗ trợ:

- schema cũ: `evaluation` phẳng + `submission_tests`
- schema mới: `evaluation.hard_filters`, `data_requirements`, `diversity`, `ranking`, `robustness`

## Quick start on Windows

Use Python 3.11+ only. If `python main.py ...` fails with `ModuleNotFoundError: No module named 'yaml'` or similar, VS Code is usually pointing at the wrong interpreter.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
python main.py --help
```

In VS Code:

- `Ctrl+Shift+P`
- `Python: Select Interpreter`
- choose `${workspaceFolder}\\.venv\\Scripts\\python.exe`

If you run `python main.py` without a command, the CLI will default to `run-full-pipeline`.

## CLI examples

```bash
python -m pip install -e .[dev]

python main.py load-data --config config/dev.yaml
python main.py generate --config config/dev.yaml --count 100
python main.py evaluate --config config/dev.yaml
python main.py top --config config/dev.yaml --limit 10
python main.py report --config config/dev.yaml --limit 5
python main.py memory-top-patterns --config config/dev.yaml --limit 10
python main.py lineage --config config/dev.yaml --alpha-id <alpha_id>
python main.py mutate --config config/dev.yaml --from-top 20 --count 50
python main.py run-full-pipeline --config config/dev.yaml
```

Console script mới:

```bash
wq-tool run-full-pipeline --config config/research.yaml
```

CSV exports mặc định:

- sau `generate`: `outputs/generated_alphas.csv`
- sau `evaluate`: `outputs/evaluated_alphas.csv`
- alpha đã được chọn: `outputs/selected_alphas.csv`

Mỗi lần chạy cũng sẽ có bản theo `run_id`, ví dụ `outputs/generated_alphas_<run_id>.csv`.

## Storage schema overview

Các bảng chính:

- `runs`: run metadata, config snapshot, profile, dataset fingerprint, regime key
- `alphas`: alpha candidates và generation metadata
- `alpha_parents`: lineage edge table
- `metrics`: train/validation/test metrics
- `selections`: alpha đã chọn + ranking rationale
- `submission_tests`: từng test robustness/subuniverse/ladder
- `simulation_cache`: cache theo simulation signature
- `alpha_history`: evaluation history cho adaptive learning
- `alpha_diagnoses`: fail/success tags và hints
- `alpha_patterns`, `alpha_pattern_membership`: pattern memory/gene scoring

## Development và testing

```bash
python -m pip install -e .[dev]
pytest -q
ruff check .
black --check .
mypy .
```

Pytest đã có smoke/integration tests cho parser, operators, metrics, filtering, critic, memory, guided generation, và pipeline end-to-end.

## Roadmap gần

- tách evaluation/generation logic sâu hơn khỏi repository helpers
- mở rộng loader cho parquet và universe lớn hơn
- factor packs và operator families nhiều hơn
- richer experiment comparison UI / notebook helpers
- distributed evaluation và worker-based generation
- adapters sang external execution/research environments

## Tài liệu chi tiết hơn

- [Architecture](docs/architecture.md)
- [Pipeline](docs/pipeline.md)
- [Configuration](docs/configuration.md)
- [Development](docs/development.md)
