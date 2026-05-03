# Code Quality Report For Claude

Generated: 2026-05-01

Muc tieu tai lieu nay: dua cho Claude mot ban tom tat chat luong code, rui ro ky thuat, ket qua kiem chung va cau hoi review cu the cho repo `wq-tool`. Tai lieu nay nen gui kem file zip source code review va `docs/current_brain_search_space_for_claude.md`.

## 1. Executive Summary

Repo la cong cu nghien cuu alpha cho WorldQuant BRAIN, tap trung vao vong lap:

1. nap field/operator catalog
2. sinh bieu thuc alpha tu genome, recipe, mutation, quality polish
3. validate/dedup/prefilter
4. submit len BRAIN API/manual backend
5. poll/normalize/persist ket qua that
6. hoc tu outcome that de dieu chinh search space va parent selection

Trang thai chat luong hien tai:

- Kien truc da co phan lop ro: domain/core/generator/service/storage/adapter/cli.
- Test suite kha rong: 408 tests collected.
- Nhom test lien quan thay doi moi chay pass: 20/20.
- Full test suite chua xac nhan pass trong lan kiem tra nay vi timeout sau khoang 184 giay.
- Ruff hien con 142 lint errors, trong do 121 loi co the auto-fix; phan lon la import sorting, pyupgrade, unused imports, va mot so style issue.
- Rui ro lon nhat nam o cac module service/storage qua dai, nhieu heuristic config, va logic van hanh BRAIN API phu thuoc trang thai that cua DB/session/service.

Ket luan ngan: codebase co huong kien truc tot va co nhieu test cho domain quan trong, nhung can review ky cac duong van hanh BRAIN, schema migration, search-space filtering va scoring heuristic truoc khi tin cay de chay dai han.

## 2. Review Scope

Nen review cac nhom source sau:

- `adapters/`: backend BRAIN manual/API.
- `alpha/`: parser, AST, validator.
- `generator/`: genome, grammar, mutation, repair, novelty.
- `services/`: orchestration, batch generation, BRAIN service, search-space filter, quality polish, recipe guided generation.
- `storage/`: SQLite schema, repositories, result/history stores.
- `core/`: scoring, BRAIN checks, config facade.
- `config/`: dataclass config va YAML profile.
- `tests/`: behavior coverage.
- `docs/architecture.md` va `docs/current_brain_search_space_for_claude.md`.

Khong nen dua vao review cac artifact runtime:

- `.git/`, `.venv/`, cache dirs
- SQLite DB files
- `outputs/`, `progress_logs/`, `backups/`
- PDF documents
- full `inputs/` snapshots
- real secrets/session files

## 3. Current Working Set

`git diff --stat` tai thoi diem lap bao cao:

- 25 tracked files changed
- 896 insertions, 44 deletions
- 7 untracked source/test/doc files lien quan

Tracked files dang thay doi:

- `adapters/brain_api_adapter.py`
- `cli/app.py`
- `cli/commands/service_status.py`
- `config/brain_full.yaml`
- `config/models/adaptive.py`
- `core/quality_score.py`
- `domain/brain.py`
- `domain/simulation.py`
- `services/brain_batch_service.py`
- `services/brain_service.py`
- `services/quality_polisher.py`
- `services/recipe_guided_generator.py`
- `services/search_space_filter.py`
- `services/selection_service.py`
- `services/status_service.py`
- `storage/brain_result_store.py`
- `storage/repositories/alpha_repository.py`
- `storage/repositories/recipe_repository.py`
- `storage/sqlite.py`
- related tests

Untracked files lien quan:

- `core/brain_checks.py`
- `cli/commands/backfill_brain_checks.py`
- `cli/commands/diagnose_fields.py`
- `tests/test_brain_checks.py`
- `tests/test_backfill_brain_checks.py`
- `tests/test_field_diagnostics.py`
- `docs/current_brain_search_space_for_claude.md`

Untracked `dist/` chi la artifact zip, khong nen review nhu source.

## 4. Architecture Assessment

Kien truc tong the dung huong:

- Domain models nam trong `domain/` va duoc re-export de giu backward compatibility.
- `SimulationAdapter` tao boundary giua orchestration noi bo va backend BRAIN/manual.
- `BrainService` normalize ket qua external ve schema noi bo.
- `ClosedLoopService` va service runtime khong can biet backend la manual hay API.
- Storage co facade `SQLiteRepository` va repository con theo nhom.
- Config duoc tach thanh dataclass group trong `config/models/`.
- CLI modules chu yeu mong, dispatch sang service.

Diem manh:

- Co contract ro: BRAIN result la source of truth, khong fake simulation de day closed-loop.
- Co traceability model day du: alpha, parent, batch, submission, result, round, history.
- Co restart-safety concept: pending submissions, ambiguous submitting batch, runtime lock, batch recovery.
- Co memory learning theo region/regime va pattern/case memory.
- Co pre-validation, dedup, repair, diversity, crowding va multi-objective selection.

Diem can Claude soi ky:

- `BrainService` vua submit, poll, timeout, recover, normalize, persist, prune invalid field metadata. Day la module trung tam co blast radius lon.
- `SQLiteRepository` va `storage/sqlite.py` van la facade/schema hub lon; migration additive can duoc review de tranh break DB cu.
- `ServiceWorker`/service runtime co nhieu trang thai operational: auth cooldown, persona confirmation, pending capacity, ambiguous resubmit.
- `search_space_filter` hien co nhieu heuristic multiplier va lane-specific filtering; can review de tranh overfit hoac lam search space qua hep.

## 5. Runtime/Search Space Snapshot

Theo `docs/current_brain_search_space_for_claude.md`:

- BRAIN profile: `USA`, `TOP3000`, delay `1`, neutralization `SUBINDUSTRY`, decay `5`, truncation `0.01`.
- Batch size: `10`.
- Max pending jobs: `10`.
- Allowed operators: 30.
- Current eligible field registry: 834 fields.
- Numeric/matrix fields dung lam alpha input: 164.
- Group/vector fields dung lam group key: 670.
- Runtime sample fields: 14.
- Catalog-only fields: 820.

Rui ro search space:

- `allow_catalog_fields_without_runtime=True` nghia la generator co the dung nhieu field chi co metadata catalog, khong co local runtime sample.
- 820/834 fields la catalog-only, nen local validation chi bat duoc syntax/type/metadata level, khong dam bao BRAIN runtime chap nhan hoac co coverage tot.
- Field usage gan day bi nghieng manh ve Analyst/Fundamental Analyst Estimates; can review diversification va risk of repeatedly exploiting stale/low-coverage fields.
- Lane allowlist theo `quality_polish`, `recipe_guided`, `fresh` giup giam invalid combinations, nhung cung co nguy co loai bo operator tot neu heuristic qua chat.

## 6. Verification Results

Commands da chay:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_brain_checks.py tests/test_backfill_brain_checks.py tests/test_field_diagnostics.py tests/test_quality_score.py tests/test_search_space_filter.py tests/test_service_status_command.py
```

Ket qua:

- 20 passed in 9.50s.

```powershell
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

Ket qua:

- 408 tests collected in 8.56s.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Ket qua:

- Timeout sau khoang 184s.
- Khong co ket luan full suite pass/fail trong lan nay.

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

Ket qua:

- 142 errors.
- 121 fixable voi `--fix`.
- Loi chinh: import sorting (`I001`), pyupgrade (`UP017`, `UP035`, `UP037`), unused imports/vars (`F401`, `F841`), ambiguous variable (`E741`), module import not at top (`E402`), f-string khong co placeholder (`F541`).

Source size:

- 238 Python files.
- 57,585 lines Python trong cac thu muc source/test chinh.
- 50 test files.

## 7. High-Risk Modules By Size

Module lon nhat nen review vi rui ro complexity:

| File | Lines | Risk |
|---|---:|---|
| `services/recipe_guided_generator.py` | 1727 | Nhieu bucket/field-pair/yield heuristic, de co edge case ve budget va duplicate retry |
| `services/kpi_report_service.py` | 1317 | Query/report aggregation phuc tap |
| `services/selection_service.py` | 1198 | Ranking/scoring/pre-screen central, de sai weighting |
| `storage/alpha_history.py` | 1166 | Memory persistence va case/pattern rebuild phuc tap |
| `services/quality_polisher.py` | 1129 | Expression transform/variant generation, de tao invalid expression neu AST cleanup sai |
| `services/brain_batch_service.py` | 1101 | Budget mix, batch prep, source metrics, orchestration |
| `services/brain_service.py` | 1086 | Submit/poll/recover/timeout/normalize/persist |
| `storage/sqlite.py` | 953 | Schema migration va DB compatibility |
| `generator/engine.py` | 937 | Candidate generation + validation path |
| `adapters/brain_api_adapter.py` | 789 | Auth/API/retry/persona/session parsing |
| `services/search_space_filter.py` | 799 | Field/operator priors, penalties, lane pools |

Claude nen uu tien review file tren truoc vi thay doi nho trong day co the anh huong toan pipeline.

## 8. Strengths

### 8.1 Test Coverage Behavior Tot O Core Paths

Test suite co nhieu case cho:

- parser/validator nesting rules
- operator registry va group operator semantics
- generation duplicate/redundancy/failure classification
- search-space filter priors and penalties
- quality polish variants
- recipe guided budget and field rotation
- BrainService retry/timeout/recover
- service mode runtime lock, cooldown, persona flow, ambiguous resubmit
- config backward compatibility
- SQLite migration

Diem nay lam repo co kha nang refactor an toan neu test suite duoc chay day du trong CI/local.

### 8.2 Operational Traceability Tot

Schema va service design co y thuc audit:

- raw BRAIN result duoc giu lai
- normalized result co status/metrics/checks/rejection
- submissions va batches tach rieng
- alpha history va case memory luu lineage/outcome
- service runtime luu heartbeat/status/recent counts

Day la diem quan trong vi BRAIN API co nhieu trang thai khong dong bo: pending, timeout, rejected, persona auth, concurrent limit, ambiguous submit.

### 8.3 Search-Space Control Da Duoc Lam Ro

Repo khong chi random generate expression; no co:

- allowed operators
- typed field registry
- group key validation
- lane-specific operator allowlists
- field diagnostic multipliers
- winner priors
- hard-fail and warning multipliers
- recipe/quality/fresh source budgets

Huong nay dung voi bai toan toi uu alpha trong search space lon.

## 9. Key Risks And Review Questions

### 9.1 Brain Check Semantics

Thay doi moi them `core/brain_checks.py`, check summary columns, submit-ready derivation va search-space penalties theo checks.

Claude nen review:

- Mapping raw BRAIN `is`/checks payload co dung voi cac shape response thuc te khong?
- Near-miss outcome checks co bi phat qua nang hoac qua nhe khong?
- Structural risk blocker co nen hard-block field/operator hay chi demote?
- `derive_submit_ready` co truong hop false positive/false negative nao khong?
- Backfill idempotency va migration columns co an toan voi DB cu khong?

### 9.2 Quality Score Calibration

`core/quality_score.py` dung multi-objective heuristic: reward Sharpe/Fitness/eligible, penalty turnover/drawdown/rejection/checks.

Claude nen review:

- Heuristic co lam chim alpha "near miss" nhung co kha nang sua khong?
- Operational timeout sau downtime da duoc loai khoi penalty alpha signal dung chua?
- Check penalty co double-count voi rejection penalty khong?
- Score range/coefficient co on dinh khi metric missing/null khong?

### 9.3 Search-Space Filter Overfitting

`services/search_space_filter.py` dang dung nhieu multiplier:

- profile mismatch
- local validation field penalty
- completed result priors
- hard fail/warning checks
- winner prior
- weak field/operator demotion
- lane field caps/min counts
- lane operator allowlists

Claude nen review:

- Multipliers co the nhan chong len nhau lam field tot bi dua ve gan zero khong?
- Winner prior co overfit vao sample nho khong?
- Min support/cap co hop ly voi current 135 submitted rows after hotfix khong?
- Field-level hard fail co nen phan biet do field, operator, expression structure, hay BRAIN temporary issue khong?
- Lane caps `recipe_guided=60`, `fresh=60`, min `30` co lam mat diversity khi catalog co 164 numeric fields khong?

### 9.4 BRAIN API Adapter And Service Reliability

`adapters/brain_api_adapter.py` va `services/brain_service.py` la duong critical.

Claude nen review:

- Retry/backoff/rate limit co the duplicate submit khong?
- Persona/auth flow co race condition voi service loop khong?
- Session file reload/save co leak credential/session khong?
- `except Exception` co dang swallow loi can fail-fast khong?
- `poll_batch`, `recover_jobs`, `timeout_pending_batch_jobs` co consistent status transition khong?
- Ambiguous submitting batch resubmit policy co replay guard du chong duplicate simulation khong?

### 9.5 SQLite Schema And Migration

`storage/sqlite.py` co schema/migration central va cac store/repository phu thuoc nhieu.

Claude nen review:

- Additive migration co idempotent voi DB cu khong?
- Columns moi cho quality score/checks co default/backfill hop ly khong?
- Indexes co du cho recent-window queries trong search-space filter/status report khong?
- Query chunking co tranh SQLite variable limit khong?
- Long transaction/batch update co nguy co lock service runtime DB khong?

### 9.6 Test Suite Runtime And CI Signal

Full suite timeout sau 184s, trong khi collect la 408 tests.

Claude nen review:

- Test nao cham nhat va co can marker `slow` khong?
- Co integration tests dang dung DB/service state that khong?
- Co nen chia quick unit suite va slow service suite khong?
- Co can them CI command rieng cho changed-path tests khong?

### 9.7 Lint Debt

Ruff 142 errors la no ky thuat ro rang.

Claude nen review:

- Co nen auto-fix 121 loi truoc khi review logic khong?
- Nhung loi con lai co bug thuc te khong: `F401`, `F841`, `E741`, `E402`, `F541`.
- Co file nao dang import sys.path hack trong `tools/` nen exclude hay chuan hoa package execution?

## 10. Suggested Claude Review Plan

De Claude review hieu qua, yeu cau no di theo thu tu:

1. Review architecture boundary: adapter/service/storage/generator co bi leak responsibility khong.
2. Review changed files first, dac biet `core/brain_checks.py`, `core/quality_score.py`, `services/search_space_filter.py`, `services/brain_service.py`, `storage/sqlite.py`.
3. Review critical runtime failure modes: duplicate submit, auth/persona, timeout, stale pending jobs, DB migration.
4. Review heuristic correctness: quality score, winner prior, check penalty, source budget.
5. Review test gaps: test nao con thieu de bat bug high-risk.
6. Output findings theo severity va file/path cu the.

## 11. Prompt To Paste Into Claude

```text
<context>
I am sending you a Python project named wq-tool. It is a WorldQuant BRAIN alpha research tool. The core loop is: build alpha expressions, validate/deduplicate, submit to BRAIN API/manual backend, poll real BRAIN results, persist traceability, and learn from real outcomes to guide the next search rounds.

Use the attached source zip plus these two reports:
- docs/code_quality_report_for_claude.md
- docs/current_brain_search_space_for_claude.md

Important constraints:
- Review only what is present in the supplied files. If you are uncertain, say so explicitly.
- Do not invent runtime facts, BRAIN API behavior, or database contents not shown in the files.
- Prioritize correctness, operational safety, duplicate-submission risk, DB migration safety, and search-space/scoring quality.
- Only discuss refactors that materially reduce risk. Do not suggest broad rewrites unless there is a clear correctness or maintainability reason.
</context>

<task>
Perform a senior-engineer code quality review of this project. Focus especially on:
1. architecture boundaries between generator, services, adapters, and storage
2. changed/current-risk modules: core/brain_checks.py, core/quality_score.py, services/search_space_filter.py, services/brain_service.py, adapters/brain_api_adapter.py, storage/sqlite.py
3. BRAIN result normalization, submit-ready/check semantics, timeout/recovery behavior, and duplicate submission prevention
4. quality scoring and search-space filtering heuristics
5. SQLite migration/idempotency risks
6. test coverage gaps and slow/full-suite verification issues
7. lint/style issues that may hide real bugs
</task>

<output_format>
Return a structured review with these sections:

1. Top Findings
- Ordered by severity: Critical, High, Medium, Low.
- For each finding include: file/path, why it matters, concrete failure scenario, recommended fix, and test to add/update.

2. Architecture Assessment
- What is strong.
- Where responsibilities are blurred.
- Which refactors are worth doing now versus later.

3. Operational Risk Review
- Duplicate submission risk.
- Auth/persona/session risk.
- Timeout/recovery risk.
- DB migration/runtime-lock risk.

4. Search Quality Review
- Field/operator filtering risks.
- Winner prior and check-penalty calibration.
- Potential overfitting or over-pruning.

5. Test Plan
- Minimal tests needed before trusting this in long-running service mode.
- Slow tests that should be split or marked.

6. Final Recommendation
- Whether this is safe for continued service-mode experimentation.
- What must be fixed before running unattended for long periods.
</output_format>
```

## 12. Files Claude Should Inspect First

Priority 1:

- `core/brain_checks.py`
- `core/quality_score.py`
- `services/search_space_filter.py`
- `services/brain_service.py`
- `adapters/brain_api_adapter.py`
- `storage/sqlite.py`
- `storage/brain_result_store.py`
- `storage/repositories/alpha_repository.py`
- `services/status_service.py`

Priority 2:

- `services/brain_batch_service.py`
- `services/quality_polisher.py`
- `services/recipe_guided_generator.py`
- `services/selection_service.py`
- `config/models/adaptive.py`
- `config/brain_full.yaml`
- `cli/commands/backfill_brain_checks.py`
- `cli/commands/diagnose_fields.py`

Priority 3:

- `tests/test_brain_checks.py`
- `tests/test_quality_score.py`
- `tests/test_search_space_filter.py`
- `tests/test_brain_integration.py`
- `tests/test_service_mode.py`
- `tests/test_sqlite_migration.py`

## 13. My Main Questions For Claude

1. Co bug nao trong logic normalize BRAIN checks co the lam alpha bi danh gia sai submit-ready khong?
2. Search-space filter co dang qua aggressive khi demote field/operator dua tren sample it hoac operational failures khong?
3. Quality score co double-count rejection/check failures khong?
4. Recovery/resubmit flow co the duplicate submission khi API timeout/limit/persona interrupt khong?
5. SQLite migrations moi co idempotent va backward-compatible voi DB cu khong?
6. Module nao nen tach nho truoc de giam risk ma khong lam refactor qua rong?
7. Nhung test nao can them ngay de bat cac bug nghiem trong nhat?

