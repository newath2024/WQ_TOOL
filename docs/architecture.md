# Architecture

## Mục tiêu

Repo được tổ chức để CLI mỏng, service orchestration rõ ràng, và core research logic nằm ở các module nhỏ hơn có thể test độc lập.

## Layering

### CLI

- `cli/app.py` dựng parser và dispatch command
- `cli/commands/*` chỉ parse args, gọi service, format output

### Services

- services điều phối workflow cấp command
- services không chứa logic numerical chuyên sâu nếu đã có ở lower-level modules
- services trả typed result dataclasses để CLI không phải xử lý raw repository rows

### Core logic

- `alpha/`: parse, validate, evaluate expressions
- `features/`: reusable operators/transforms
- `backtest/`: simulation, positions, costs, metrics
- `evaluation/`: filtering, critic, ranking, submission-style tests
- `generator/`: expression generation và adaptive mutation
- `memory/`: structural decomposition và regime-scoped adaptive scoring

### Persistence

- `storage/sqlite.py` giữ DDL + migration kiểu additive
- `storage/repository.py` là IO boundary chính
- `storage/alpha_history.py` giữ persistence/query logic riêng cho adaptive memory

## Thiết kế hiện tại

- incremental refactor, không rewrite logic cốt lõi
- backward-compatible CLI surface
- config loader chấp nhận schema cũ và schema nhóm mới
- run metadata giàu hơn để `report`, `memory`, `lineage` không phải suy luận lại từ output thủ công

## Traceability

Một alpha đã chọn nên truy được:

- run nào tạo ra nó
- config/profile nào đang dùng
- dataset fingerprint / regime key nào áp dụng
- generation mode, parent refs, lineage depth
- metrics train/validation/test
- fail/success tags và rejection reasons
- ranking rationale dùng để shortlist
