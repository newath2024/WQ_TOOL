# Architecture

## Muc tieu

Repo nay duoc chia thanh 2 lop:

- local research layer: xay search space, validate, dedup, xep hang so bo, memory support
- BRAIN integration layer: submit/simulate/collect result that, luu traceability, hoc tu outcome that

Nguyen tac quan trong:

- khong couple generator vao backend BRAIN
- khong fake BRAIN simulation bang local backtest
- BRAIN result la source of truth cho closed-loop
- manual backend va api backend cung di qua `SimulationAdapter`

## Layering

### Registry va core research logic

- `data/field_registry.py`: field metadata, runtime fields, field score
- `features/registry.py`: typed operator registry
- `alpha/`: parser, AST, validator, evaluator
- `generator/`: template generation, mutation, guided generation
- `memory/pattern_memory.py`: structural signature, pattern score, outcome learning

### BRAIN integration

- `adapters/simulation_adapter.py`: abstraction chung
- `adapters/brain_manual_adapter.py`: export/import cho workflow thu cong
- `adapters/brain_api_adapter.py`: scaffold san sang noi API that
- `services/brain_service.py`: submit, poll, normalize, persist
- `services/candidate_selection_service.py`: pre-rank, diversity, select-for-simulate, select-for-mutate
- `services/closed_loop_service.py`: vong lap generate -> BRAIN -> learn -> mutate
- `workflows/run_brain_simulation.py`: workflow submit 1 batch
- `workflows/run_closed_loop.py`: workflow nhieu round

### Persistence

- `storage/sqlite.py`: schema va additive migration
- `storage/repository.py`: main repository boundary
- `storage/submission_store.py`: submission batches + submission records + manual imports
- `storage/brain_result_store.py`: normalized BRAIN results + closed-loop summaries
- `storage/alpha_history.py`: pattern memory/history cho ca local va external outcomes

### CLI

- `cli/app.py`: parse argv, dispatch command
- `cli/commands/*`: command modules mong, khong chua business logic

## SimulationAdapter boundary

Interface chung:

```python
class SimulationAdapter:
    def submit_simulation(self, expression: str, sim_config: dict) -> dict: ...
    def get_simulation_status(self, job_id: str) -> dict: ...
    def get_simulation_result(self, job_id: str) -> dict: ...
    def batch_submit(self, expressions: list[str], sim_config: dict) -> list[dict]: ...
```

Y nghia:

- adapter chi noi voi backend external
- `BrainService` chuyen ket qua adapter ve schema noi bo
- `ClosedLoopService` khong can biet backend la manual hay api

## Normalized BRAIN result model

Noi bo he thong dung schema da chuan hoa:

```json
{
  "expression": "...",
  "job_id": "...",
  "status": "completed | failed | rejected | timeout | manual_pending",
  "region": "...",
  "universe": "...",
  "delay": 1,
  "neutralization": "...",
  "decay": 0,
  "metrics": {
    "sharpe": 1.2,
    "fitness": 0.8,
    "turnover": 0.5,
    "drawdown": 0.2,
    "returns": 0.07,
    "margin": 0.04
  },
  "submission_eligible": true,
  "rejection_reason": null,
  "raw_result": {...},
  "simulated_at": "..."
}
```

`raw_result` luon duoc giu lai de debug/audit.

## Traceability model

He thong phai tra loi duoc:

- candidate nao da duoc simulate
- submit trong batch/job nao
- submit voi config nao
- ket qua that tu BRAIN la gi
- co reject khong
- vi sao reject
- candidate nao la parent/child
- round nao chon candidate nay de mutate

Bang lien quan:

- `alphas`
- `alpha_parents`
- `submission_batches`
- `submissions`
- `brain_results`
- `manual_imports`
- `closed_loop_runs`
- `closed_loop_rounds`
- `alpha_history`

## Closed-loop lifecycle

1. generate candidate co cau truc
2. local validate va dedup
3. candidate selection policy chon top-N da dang
4. `BrainService` submit len BRAIN
5. poll/import ket qua
6. normalize va persist metrics/rejection/raw_result
7. cap nhat `alpha_history` voi `metric_source=external_brain`
8. chon parent manh dua tren BRAIN outcome
9. sinh mutation cho round tiep theo

Neu backend la `manual`:

- round se dung o trang thai `waiting_manual_results` neu chua co file ket qua import vao
- he thong khong doan/fake result de di tiep

## Memory hoc tu BRAIN

Pattern memory da hoc tren:

- template
- field
- operator
- operator family
- lookback
- wrapper
- subexpression
- rejection reason

Positive signal:

- sharpe tot
- fitness tot
- turnover chap nhan duoc
- submission eligible

Negative signal:

- rejected
- high turnover
- poor fitness
- duplicate family khong cai thien
- excessive complexity

Logic nay chu y giai thich duoc, khong phai black-box model.
