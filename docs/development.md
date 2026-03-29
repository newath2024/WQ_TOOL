# Development

## Setup

```bash
python -m pip install -e .[dev]
```

## Core commands

```bash
pytest -q
ruff check .
black --check .
mypy .
```

## Refactor guidelines

- giữ CLI surface backward-compatible
- ưu tiên additive migrations cho SQLite
- không đẩy orchestration xuống repository hoặc utility modules
- validation metrics là driver chính cho adaptive logic
- test split chỉ để audit

## Khi thêm feature mới

1. chọn layer đúng: CLI / service / core / storage
2. thêm config explicit, tránh magic number
3. persist đủ metadata để debug run sau này
4. thêm smoke test hoặc integration test nếu feature chạm workflow

## Định hướng mở rộng

- data readers mới nên đi qua `data/` + `services/data_service.py`
- generation strategies mới nên cắm vào `generator/` rồi được gọi từ generation/mutation service
- report/UI layer nên đọc từ service/query layer, không query thẳng SQLite trong presentation
