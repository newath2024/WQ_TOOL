# Project Context

Tai lieu nay la ban do tong quan de dung cho cac phien lam viec sau.
Neu can vao viec nhanh, hay doc file nay truoc, sau do moi mo cac file chi tiet.

## 1. Du an nay la gi

`WQ Tool` la mot framework research alpha theo huong WorldQuant/BRAIN.
Repo nay co 3 lop chinh:

- local research layer: nap data, build field/operator registry, sinh alpha, validate, local evaluate/backtest, memory
- BRAIN integration layer: export manual hoac goi API, poll/import ket qua, normalize result, hoc tu outcome that
- orchestration/service layer: closed-loop, candidate selection, duplicate/crowding control, 24/7 service mode, runtime lock, heartbeat

Nguyen tac quan trong:

- local evaluation/backtest khong phai source of truth cho closed-loop BRAIN
- BRAIN result moi la source of truth khi chon parent va hoc cho vong sau
- backend `manual` va `api` cung di qua abstraction `SimulationAdapter`

## 2. Quy mo repo

Snapshot luc doc repo:

- khoang 470 tracked files
- khoang 170 file Python
- 39 file test trong `tests/`

Luu y:

- mot phan lon file tracked khong phai code ma la PDF tai lieu va CSV/JSON snapshot du lieu BRAIN
- thu muc `inputs/wq_snapshots/2026-03-29/` la tap catalog/metadata lon, khong phai business logic
- thu muc `Document/` la tai lieu tham khao BRAIN, khong phai code runtime

## 3. Mental model ngan

Hay hinh dung he thong theo flow nay:

1. doc config
2. mo SQLite repository
3. load market/research context + field registry
4. build candidate tu `Genome -> AST -> normalized expression`
5. validate / dedup / crowding / selection
6. local evaluate hoac submit len BRAIN
7. persist metrics, lineage, result, memory, service state
8. neu la closed-loop thi chon parent va sinh mutation/crossover cho round tiep

Entrypoint chinh:

- `main.py`: bootstrap Python/runtime dependency
- `cli/app.py`: parser + dispatch command
- `services/runtime_service.py`: mo repository, resolve run context, init run metadata

## 4. Ban do thu muc

Thu muc quan trong nhat:

- `adapters/`: boundary ra ngoai, gom `brain_manual_adapter.py`, `brain_api_adapter.py`, `simulation_adapter.py`
- `alpha/`: parser, AST, validator, evaluator cho expression
- `backtest/`: engine local cho workflow cu
- `cli/`: command line wiring, command modules mong
- `config/`: profile config (`default`, `dev`, `research`, `strict`, `brain_full`)
- `core/`: config dataclass, logging, run context, signature utilities
- `data/`: market data loader, schema, field registry
- `docs/`: architecture/config/pipeline/development docs
- `evaluation/`: filtering, ranking, stability, submission checks
- `features/`: operator registry va transforms
- `generator/`: genome search engine, grammar, mutation, crossover, novelty, repair
- `memory/`: pattern memory va case memory
- `registry/`: field/operator registry helpers
- `services/`: orchestration va business logic cap cao
- `storage/`: SQLite schema + store/repository layer
- `tests/`: regression test cho parser, engine, service, selection, storage, CLI
- `tools/`: PowerShell/Python/JS helper scripts
- `workflows/`: wrapper workflow mong cho CLI
- `inputs/`: snapshot field catalog/operator catalog/dataset metadata tu BRAIN
- `outputs/`, `progress_logs/`, `backups/`, `TEMP/`: artifact/runtime data

## 5. Cac khoi code can nam

### 5.1 Config va runtime

- `core/config.py` la noi dinh nghia toan bo schema config bang dataclass
- Config khong chi co local pipeline ma con co `brain`, `loop`, `service`, `adaptive_generation.region_learning`, duplicate/crowding/selection
- `services/runtime_service.py` quyet dinh command nao duoc resume run cu va command nao tao run moi

Dieu nay co nghia:

- gan nhu moi hanh vi cua tool deu di qua config
- muon hieu he thong dang chay the nao, phai xem config profile truoc

### 5.2 Data va research context

- `services/data_service.py` la diem vao hop nhat de load du lieu nghien cuu
- no load OHLCV + aux data + runtime field values + field catalog + regime context
- no cung persist dataset summary, regime snapshot va run field scores vao SQLite
- co cache cho `research_context` de service/generation khong rebuild lai vo ich

Luu y:

- local sample data nam o `examples/sample_data/`
- full field catalog generation dung `inputs/wq_snapshots/...` qua `generation.field_catalog_paths`
- che do `allow_catalog_fields_without_runtime: true` cho phep generate field catalog-only de submit BRAIN, du local runtime khong co full field values

### 5.3 Alpha language va generation engine

- `alpha/parser.py`, `alpha/validator.py`, `alpha/evaluator.py` xu ly expression language
- `generator/engine.py` dinh nghia `AlphaCandidate` va generation engine co validation, dedup, diversity, repair
- `generator/guided_generator.py` them lop guided generation dua tren pattern memory + case memory
- `generator/genome_builder.py`, `generator/grammar.py`, `generator/mutation_policy.py`, `generator/crossover.py`, `generator/repair_policy.py` la tim cua bo may sinh alpha

Mental model dung:

- repo nay khong con la template-string generator don gian
- candidate moi la mot genome co metadata cau truc
- expression string van duoc giu lam external contract voi validator, CLI va BRAIN

### 5.4 Memory va hoc hoi

- `memory/pattern_memory.py`: hoc tren structural signature, family, field, operator, regime
- `memory/case_memory.py`: hoc o muc case/objective/mutation-path phong phu hon
- `storage/alpha_history.py`: persistence cho memory/history/case data

Diem quan trong:

- memory da region-aware
- local scope va global prior scope duoc blend qua config
- parent pool mac dinh uu tien local region/regime, khong tron tat ca vao mot memory pool global don

### 5.5 Candidate selection

- `services/candidate_selection_service.py` la orchestration layer cho pre-screen, duplicate, crowding, score va select
- `services/duplicate_service.py` lo exact/structural/cross-run duplicate
- `services/crowding_service.py` phat hien collapse theo family/motif/operator path/lineage/history
- `services/selection_service.py` score va chon truoc-sim va sau-sim
- `services/multi_objective_selection.py` dung non-dominated sorting + crowding distance

Thuc te:

- truoc khi vao BRAIN, candidate da di qua nhieu lop loc
- selection khong chi rank theo score ma con giu da dang cau truc

### 5.6 BRAIN integration

- `services/brain_service.py` submit batch, poll batch, import/normalize/persist result
- `adapters/brain_manual_adapter.py` dung cho export/import CSV thu cong
- `adapters/brain_api_adapter.py` chua logic auth/session/Persona/rate-limit cho API mode
- `services/session_manager.py` va `services/notification_manager.py` quan ly session + Persona flow

Backend:

- `manual`: xuat file candidate, user tu submit len BRAIN, sau do import ket qua
- `api`: tool tu submit/poll neu session hop le

Can nho:

- day la boundary nhay cam nhat cua he thong
- khi co van de auth, Persona, throttle, duplicate submission, hay nhin o `adapters/` va `services/brain_service.py`

### 5.7 Closed-loop va service mode

- `services/closed_loop_service.py` chay vong `generate -> select -> BRAIN -> learn -> mutate`
- `services/service_worker.py` xu ly 1 tick service mode
- `services/service_runner.py` va `workflows/run_service.py` bao boi process loop
- `services/runtime_lock.py` dam bao moi DB/profile chi co 1 instance service so huu lease

Service mode khong phai la wrapper don quanh `run-closed-loop`.
No co logic rieng:

- resume pending submissions
- recover batch `submitting`
- quarantine/resubmit/fail batch mo ho tuy config
- heartbeat/cooldown/auth wait state

## 6. Workflow chinh

### 6.1 Local workflow

Muc dich:

- debug parser/evaluator
- smoke test tren sample dataset
- local screening truoc khi submit BRAIN

Lenh tieu bieu:

```bash
python main.py generate --config config/dev.yaml
python main.py evaluate --config config/dev.yaml
python main.py run-full-pipeline --config config/dev.yaml
```

### 6.2 Manual BRAIN workflow

Muc dich:

- dung khi muon su dung BRAIN nhu source of truth nhung submit thu cong

Flow:

1. sync field/operator catalog neu can
2. generate candidate
3. export CSV
4. submit thu cong len BRAIN
5. import ket qua CSV ve repo
6. closed-loop hoc tiep tu external outcome

Lenh tieu bieu:

```bash
python main.py sync-field-catalog --config config/dev.yaml
python main.py export-brain-candidates --config config/dev.yaml
python main.py import-brain-results --config config/dev.yaml --path outputs/brain_manual/manual_results.csv
python main.py run-closed-loop --config config/dev.yaml
```

### 6.3 API closed-loop workflow

Muc dich:

- tu dong submit/poll/import neu API/session hop le

Lenh tieu bieu:

```bash
python main.py brain-login --config config/dev.yaml
python main.py run-brain-sim --config config/dev.yaml
python main.py run-closed-loop --config config/dev.yaml
```

### 6.4 24/7 service workflow

Muc dich:

- foreground service de duoc scheduler/goi lap lai tren 1 may

Lenh tieu bieu:

```bash
python main.py run-service --config config/dev.yaml
python main.py service-status --config config/dev.yaml
```

Behavior:

- doc `service_runtime`
- acquire/renew DB lease
- ensure session
- poll pending jobs
- hoan tat batch thi learn vao memory
- neu khong co pending thi tao batch moi
- ghi heartbeat

## 7. SQLite la source of truth operational

Boundary DB:

- `storage/sqlite.py`: DDL + additive migration/backfill
- `storage/repository.py`: repository cap cao
- `storage/submission_store.py`, `storage/brain_result_store.py`, `storage/service_runtime_store.py`: store chuyen biet

Bang quan trong:

- `runs`: metadata cua moi run
- `alphas`, `alpha_parents`: candidate va lineage
- `metrics`, `submission_tests`, `selections`: local evaluation layer
- `alpha_history`, `alpha_patterns`, `alpha_pattern_membership`, `alpha_cases`: memory/history layer
- `field_catalog`, `run_field_scores`: field metadata va score
- `submission_batches`, `submissions`, `brain_results`, `manual_imports`: BRAIN traceability layer
- `closed_loop_runs`, `closed_loop_rounds`: closed-loop orchestration layer
- `service_runtime`: state cua 24/7 service
- `alpha_duplicate_decisions`, `alpha_crowding_scores`, `round_stage_metrics`, `alpha_selection_scores`, `mutation_outcomes`, `regime_snapshots`: diagnostic/selection/explainability layer

Neu can tra loi cau hoi "alpha nay den tu dau, da submit chua, ket qua ra sao, co duoc mutate tiep khong", cau tra loi thuong nam trong SQLite truoc tien.

## 8. Config profiles dang co

- `config/default.yaml`: profile mac dinh, gan voi `research`
- `config/research.yaml`: profile local research nghiem tuc hon sample dev
- `config/dev.yaml`: profile nhe hon, batch nho, timeout ngan, hop cho test nhanh
- `config/strict.yaml`: local screening chat hon truoc khi review alpha
- `config/brain_full.yaml`: profile quan trong neu muon generate bang full field catalog snapshot va chay service/API mode

Rule of thumb:

- neu dang debug logic: bat dau tu `config/dev.yaml`
- neu dang muon nghien cuu local nghiem tuc hon: xem `config/research.yaml` hoac `config/strict.yaml`
- neu dang nham den full BRAIN catalog/service: xem `config/brain_full.yaml`

## 9. Du lieu va artifact can biet

Nguon du lieu/code:

- `examples/sample_data/`: sample OHLCV + metadata de smoke test
- `inputs/wq_snapshots/2026-03-29/`: field catalog/operator catalog/dataset snapshot tu BRAIN
- `Document/`: PDF docs ve BRAIN/operator/data/concepts

Artifact runtime:

- `outputs/`: candidate export, manual import, API session file, local output CSV
- `progress_logs/`: progress event logs
- `backups/`: snapshot backup
- `*.sqlite3`: DB runtime thuc te

DB de y trong workspace:

- `wq_tool.sqlite3`
- `dev_wq_tool.sqlite3`
- co them mot so ban copy DB lon de nghien cuu/backup

Khi lam viec, phai phan biet ro:

- source code
- config profile
- input snapshot
- runtime artifact / operational database

## 10. File nen mo truoc khi sua code

Neu can orientation nhanh:

1. `docs/project_context.md`
2. `README.md`
3. `docs/architecture.md`
4. config dang dung trong `config/*.yaml`
5. file orchestration lien quan:
   - `cli/app.py`
   - `services/runtime_service.py`
   - `services/data_service.py`
   - `services/closed_loop_service.py`
   - `services/brain_service.py`
   - `services/service_worker.py`
   - `storage/repository.py`
   - `storage/sqlite.py`

Neu can sua generation:

- `generator/engine.py`
- `generator/guided_generator.py`
- `generator/genome_builder.py`
- `generator/mutation_policy.py`
- `generator/grammar.py`
- `alpha/validator.py`

Neu can sua selection:

- `services/candidate_selection_service.py`
- `services/selection_service.py`
- `services/duplicate_service.py`
- `services/crowding_service.py`

Neu can sua BRAIN/service:

- `adapters/brain_api_adapter.py`
- `adapters/brain_manual_adapter.py`
- `services/session_manager.py`
- `services/notification_manager.py`
- `services/service_runner.py`
- `services/service_scheduler.py`

## 11. Cach giao tiep hieu qua ve repo nay

Khi bat dau mot task moi, nen lam ro 4 diem:

1. dang dung config nao
2. backend dang la `manual` hay `api`
3. muc tieu la local evaluation hay external BRAIN result
4. DB/run nao la source of truth can doc

Neu khong ro 4 diem nay, rat de sua sai lop he thong.

## 12. Cac hieu nham de tranh

- dung nham local backtest thanh "ket qua that" cua BRAIN
- nham `examples/sample_data/` la du lieu san xuat
- bo qua `brain_full.yaml` khi muon generate tu full field catalog
- sua service mode nhu the no chi la wrapper quanh `run-closed-loop`
- xem CSV export/import la source of truth thay vi SQLite
- bo qua logic region-aware learning khi phan tich memory/mutation

## 13. Gioi han hien tai

- backend `manual` van can import ket qua thu cong truoc khi closed-loop hoc day du
- integration API la vung nhay cam, de bi anh huong boi auth/session/Persona/rate-limit/endpoint behavior
- service mode hien tai theo mo hinh single-machine lease lock, chua phai distributed coordinator
- repo chua nhe: co nhieu snapshot/DB lon, nen moi task can phan biet ro code va artifact

## 14. Tom tat mot cau

Day la mot he thong research alpha theo huong WorldQuant/BRAIN, trong do generator genome-based + region-aware memory + selection/diversity layer tao candidate, BRAIN cung cap ket qua truth, va SQLite giu toan bo traceability cho closed-loop va service mode.
