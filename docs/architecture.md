# Architecture

> Normative note: lifecycle, orchestration, timeout, and recovery contracts are defined in `docs/alpha_pipeline_spec.md`. This file remains an architecture overview and does not override the execution spec.

## Muc tieu

Repo nay duoc chia thanh 3 lop:

- local research layer: xay genome search space, render expression, validate, dedup, xep hang so bo, memory support
- BRAIN integration layer: submit/simulate/collect result that, luu traceability, hoc tu outcome that
- orchestration layer: multi-objective selection, diversity preservation, service loop, va closed-loop control

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
- `generator/genome.py`: genome dataclass cho feature/transform/horizon/wrapper/regime/turnover/complexity genes
- `generator/grammar.py`: motif grammar render `Genome -> AST -> expression`
- `generator/genome_builder.py`: random, exploit, novelty, memory-guided genome construction
- `generator/mutation_policy.py`: 5 mutation modes `exploit_local`, `structural`, `crossover`, `novelty`, `repair`
- `generator/crossover.py`: homologous gene-level crossover
- `generator/novelty.py`: novelty scoring tren structural distance
- `generator/repair_policy.py`: bounded repair pass cho invalid/high-cost candidates
- `memory/pattern_memory.py`: structural signature, pattern score, low-level outcome learning
- `memory/case_memory.py`: richer case memory theo regime/family/mutation-path/objectives

### BRAIN integration

- `adapters/simulation_adapter.py`: abstraction chung
- `adapters/brain_manual_adapter.py`: export/import cho workflow thu cong
- `adapters/brain_api_adapter.py`: scaffold san sang noi API that
- `services/brain_service.py`: submit, poll, normalize, persist
- `services/candidate_selection_service.py`: objective prediction, diversity filtering, select-for-simulate, select-for-mutate
- `services/multi_objective_selection.py`: non-dominated sorting + crowding distance
- `services/diversity_manager.py`: anti-collapse caps va exploration quota
- `services/closed_loop_service.py`: vong lap generate -> BRAIN -> learn -> mutate
- `services/session_manager.py`: non-interactive auth/session refresh cho service mode
- `services/notification_manager.py`: Persona terminal + email notification
- `services/runtime_lock.py`: DB lease lock cho single-instance service
- `services/heartbeat_reporter.py`: heartbeat, counters, last success/error
- `services/service_scheduler.py`: quyet dinh sleep interval theo state
- `services/service_worker.py`: xu ly mot service tick end-to-end
- `services/service_runner.py`: vong doi process + signal handling + restart-safe orchestration
- `workflows/run_brain_simulation.py`: workflow submit 1 batch
- `workflows/run_closed_loop.py`: workflow nhieu round
- `workflows/run_service.py`: workflow foreground service mode

### Persistence

- `storage/sqlite.py`: schema va additive migration
- `storage/repository.py`: main repository boundary
- `storage/submission_store.py`: submission batches + submission records + manual imports
- `storage/brain_result_store.py`: normalized BRAIN results + closed-loop summaries
- `storage/alpha_history.py`: pattern memory/history + case memory cho ca local va external outcomes

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
- `alpha_cases`
- `submission_batches`
- `submissions`
- `brain_results`
- `manual_imports`
- `closed_loop_runs`
- `closed_loop_rounds`
- `alpha_history`
- `service_runtime`

## Closed-loop lifecycle

1. genome builder tao genome moi hoac genome-guided tu memory
2. motif grammar render `Genome -> AST -> normalized expression`
3. validator/prefilter/repair loai hoac sua candidate khong hop le
4. candidate selection policy xep hang bang multi-objective scoring
5. diversity manager ap hard caps theo family, field category, horizon, operator path, va exploration quota
6. `BrainService` submit len BRAIN
7. poll/import ket qua
8. normalize va persist metrics/rejection/raw_result
9. cap nhat `alpha_history` voi `metric_source=external_brain`
10. cap nhat `alpha_cases` va pattern membership theo motif/family/path/mutation mode/regime
11. chon parent manh dua tren BRAIN outcome + diversity
12. sinh mutation/crossover/repair cho round tiep theo

Neu backend la `manual`:

- round se dung o trang thai `waiting_manual_results` neu chua co file ket qua import vao
- he thong khong doan/fake result de di tiep

## Service mode lifecycle

`run-service` khong goi `run-closed-loop` trong vong lap. No dung continuous batch mode:

1. boot va resolve `service_run_id`
2. acquire DB lease lock
3. ensure session qua `SessionManager`
4. recover batch `submitting`
5. neu co pending jobs thi chi poll/update/persist/learn
6. neu khong co pending jobs thi tao batch moi, submit, persist ngay tung job
7. ghi heartbeat vao `service_runtime`
8. sleep theo `ServiceScheduler`

Restart safety:

- service co the resume pending jobs tu `submissions`
- batch `submitting` du metadata se duoc recover thanh `submitted`
- batch `submitting` mo ho se bi `paused_quarantine`
- service khong auto-resubmit batch mo ho de tranh duplicate submission

## Memory hoc tu BRAIN

Pattern memory va case memory hoc tren:

- family signature
- structural signature
- field
- field family
- operator
- operator family/path
- lookback
- wrapper
- motif
- complexity bucket
- turnover bucket
- mutation mode
- rejection reason
- regime-conditioned tags

Positive signal:

- sharpe tot
- fitness tot
- turnover chap nhan duoc
- submission eligible
- robustness on dinh
- mutation mode thuc su cai thien family do

Negative signal:

- rejected
- high turnover
- poor fitness
- duplicate family khong cai thien
- excessive complexity
- motif/path combination fail lap lai
- mutation mode thuong xuyen lam xau family/regime do

Logic nay chu y giai thich duoc, khong phai black-box model.
