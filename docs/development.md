# Development

## Setup

```bash
python -m pip install -e .[dev]
```

## Core checks

```bash
pytest -q
ruff check .
black --check .
mypy .
```

## Refactor rules

- giu command local cu backward-compatible
- BRAIN integration phai di sau `SimulationAdapter`
- khong day business logic xuong CLI hoac repository thuan SQL
- khong fake external simulation
- comment/doc phai trung thuc ve local proxy logic vs BRAIN result that
- uu tien additive SQLite migrations

## Khi them feature moi

1. quyet dinh dung layer nao:
   - adapter
   - service
   - workflow
   - storage
   - docs
2. them config explicit, tranh magic number
3. persist du metadata de audit lai run sau nay
4. them test voi mock/fake adapter thay vi phu thuoc BRAIN that

## Goi y cho BRAIN API work sau nay

- khong hard-code endpoint neu chua verify
- inject transport vao `BrainApiAdapter`
- test parser/payload/retry/status mapping bang mock transport
- giu payload builder va response parser tach rieng de de review

## Quy uoc test cho BRAIN workflow

Nen cover cac nhom sau:

- adapter contract
- manual export/import flow
- result normalization
- submission/result store persistence
- retry/timeout behavior
- closed-loop round orchestration
- service mode: lock, resume, backoff, shutdown, quarantine
- memory update tu external outcomes
- regression: local `evaluate` va `run-full-pipeline` van chay

## Luu y cho `run-service`

- `run-service` la foreground process, khong tu daemonize
- dung supervisor ben ngoai de restart process khi may reboot
- test service mode nen dung fake `BrainApiAdapter` thay vi external API that
- neu them state moi cho service, uu tien persist vao `service_runtime` hoac `submissions`
- bat ky batch `submitting` mo ho nao cung phai duoc quarantine thay vi auto-resubmit
