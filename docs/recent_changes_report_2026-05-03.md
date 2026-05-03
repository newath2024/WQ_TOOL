# Bao Cao Thay Doi Gan Day - 2026-05-03

## Pham vi

Bao cao nay tong hop diff hien tai trong working tree cua repo `wq-tool`.

Snapshot truoc khi tao bao cao nay:

- File tracked da sua: 37
- Kich thuoc diff tracked: 5,647 dong them, 314 dong xoa
- File untracked: 15
- Artifact dang untracked: 4 file zip trong `dist/`
- `git diff --check`: dat, chi co canh bao line ending LF se duoc Git doi sang CRLF
- Lenh test da chay:

```powershell
py -3.12 -m pytest tests/test_quality_polisher.py tests/test_search_space_filter.py tests/test_config.py tests/test_quality_score.py tests/test_recipe_guided_generator.py tests/test_brain_checks.py tests/test_backfill_brain_checks.py tests/test_field_diagnostics.py tests/test_generator_engine.py tests/test_brain_integration.py tests/test_brain_api_auth.py tests/test_service_status_command.py -q
```

- Ket qua: 142 passed in 18.43s

## Tom tat ngan

Diff hien tai la mot dot nang cap lon cho adaptive search, khong phai mot patch nho. Cac thay doi lon nhat gom:

- Sinh bien the quality polish theo cau truc that trong `services/quality_polisher.py`
- Winner prior co guard, smoothing, recent-window va cache trong `services/search_space_filter.py`
- Parse BRAIN checks va tinh quality score dua tren check severity
- Ho tro operator-diversity generation
- Them group-relative recipe generation
- CLI cho field diagnostics va backfill BRAIN checks
- Mo rong schema/result model de luu check summary va diagnostic rows

## Thay doi theo khu vuc

### 1. Structural Quality Polish

File chinh:

- `services/quality_polisher.py`
- `config/models/adaptive.py`
- `config/brain_full.yaml`
- `tests/test_quality_polisher.py`

Noi dung thay doi:

- Them cac constant:
  - `QUALITY_POLISH_OPERATOR_ALLOWLIST`
  - `FIELD_FAMILIES`
  - operator substitution maps
  - default variant-budget percentages
  - parent structural similarity threshold
- Refactor logic sinh variant thanh cac bucket co budget rieng:
  - `surface`
  - `operator_substitution`
  - `neutralization`
  - `cross_section`
  - `composite`
  - `field_substitution`
- Them cac method structural variant:
  - `_operator_substitution_variants`
  - `_neutralization_variants`
  - `_cross_sectional_wrapper_variants`
  - `_composite_structure_variants`
  - `_field_substitution_variants`
- Them helper AST cho:
  - traverse AST
  - replace node
  - tim outer `group_neutralize`
  - depth check
  - replace field
  - tim sibling field
  - structural similarity
  - collect operator/field/lookback
  - budgeted selection
- Giu lai surface transforms cu bang cach dua vao `_surface_variants`.
- Truoc khi accept variant, co them cac check:
  - duplicate normalized expression
  - vi pham lane operator allowlist
  - qua giong parent ve structural similarity
  - fail validation qua existing candidate build path
- Them `QualityOptimizationConfig.variant_budget_percentages` voi validation va default.
- Cap nhat `brain_full.yaml` de enable cac transform structural va budget:
  - surface: 0.30
  - operator substitution: 0.20
  - neutralization: 0.15
  - cross section: 0.15
  - composite: 0.10
  - field substitution: 0.10

Tac dong hanh vi:

- Quality polish khong con bi chi phoi boi bien the chi doi lookback.
- Surface-only variants bi gioi han boi allocation, tru truong hop structural bucket khong sinh du.
- Bien the moi co kha nang thay doi structure/correlation that hon.

Tests lien quan:

- Operator substitution giu nguyen args/lookback
- Add/remove neutralization
- Composite khong vuot max depth
- Field substitution giu nguyen operator structure
- Tat ca structural variants pass validator
- Budget allocation nam trong sai so cho phep
- Surface-only khong dominate output
- Skip duplicate variants

### 2. Search-Space Winner Prior va Field/Operator Filtering

File chinh:

- `services/search_space_filter.py`
- `config/models/adaptive.py`
- `config/brain_full.yaml`
- `services/brain_service.py`
- `tests/test_search_space_filter.py`

Noi dung thay doi:

- Them cache winner-prior in-process co lock:
  - `_WINNER_PRIOR_CACHE_LOCK`
  - `_WINNER_PRIOR_CACHE`
  - `invalidate_winner_prior_cache`
- `BrainService` invalidate winner-prior cache sau khi save BRAIN results.
- Viet lai completed-result priors voi:
  - minimum completed-result guard
  - minimum winners de boost
  - minimum losers de penalty
  - Laplace smoothing
  - multiplier range co min/max
  - recent round-window cutoff
  - all-time fallback voi dampening
  - winner definition ro rang qua `is_winner_result`
  - loai operational timeout/rejection khoi prior
  - logging diagnostic cho prior stats
- Them config moi:
  - `winner_prior_min_completed`
  - `winner_prior_min_winners_for_boost`
  - `winner_prior_min_losers_for_penalty`
  - `winner_prior_laplace_k`
  - `winner_prior_multiplier_max`
  - `winner_prior_multiplier_min`
  - `winner_prior_alltime_dampen`
  - `winner_prior_cache_ttl_seconds`
  - `winner_prior_min_sharpe`
  - `winner_prior_min_fitness`
- Cap nhat `brain_full.yaml`:
  - lookback rounds: 50
  - min completed: 15
  - boost/penalty support: 3
  - Laplace k: 1.0
  - multiplier range: 0.5 den 1.5
  - all-time dampen: 0.5
  - cache TTL: 300 seconds
  - winner Sharpe threshold: 0.50
  - winner fitness threshold: 0.0
- Them floor va exploration budget:
  - `field_floor_ratio`
  - `field_floor_absolute_min`
  - `operator_floor_absolute_min`
  - `exploration_budget_pct`
- Them field diagnostic multipliers va structural-risk-aware demotions.

Tac dong hanh vi:

- Khi sample qua it, winner prior tra ve neutral thay vi overfit vao noise.
- Operational failure khong bi tinh nhu loser quality.
- History cu khong con dominate history gan day; neu phai fallback all-time thi effect bi dampen.
- Field/operator weight it bi sap ve 0 do noisy early history.

Tests lien quan:

- Du lieu khong du thi prior neutral
- Laplace smoothing tranh multiplier cuc doan
- Recent window loai old rounds
- All-time fallback co dampen
- Winner definition ro rang
- Cache hit/miss
- Operational timeout bi exclude
- Multiplier nam trong bound
- Field floor va exploration budget
- Structural-risk va diagnostic penalty

### 3. BRAIN Check Parsing va Quality Score

File chinh:

- `core/brain_checks.py` moi
- `core/quality_score.py`
- `config/models/quality.py` moi
- `config/loader.py`
- `config/models/runtime.py`
- `domain/brain.py`
- `domain/simulation.py`
- `storage/brain_result_store.py`
- `storage/sqlite.py`
- `adapters/brain_api_adapter.py`
- `services/brain_service.py`
- `tests/test_brain_checks.py`
- `tests/test_quality_score.py`
- `tests/test_brain_integration.py`
- `tests/test_brain_api_auth.py`

Noi dung thay doi:

- Them BRAIN check summary model gom:
  - raw normalized checks
  - hard fail checks
  - warning checks
  - blocking warning checks
  - pass checks
  - long/short counts
  - derived submit readiness
  - metrics chinh
- Them phan loai check:
  - outcome checks
  - robustness checks
  - structural-risk blocker checks
  - synthetic rejection checks
  - operational timeout markers
- Them helper cho:
  - summarize raw BRAIN payload
  - serialize check names
  - derive submit readiness
  - classify timeout cause
  - detect check-based rejection
  - detect structural-risk blocker
- Cap nhat `MultiObjectiveQualityScorer`:
  - nhan check summary
  - tach check-based reject va non-check reject
  - tranh double-penalty cho check-based reject
  - warning cua completed alpha bi penalty nhe hon
  - operational downtime timeout giu neutral
  - ho tro `QualityScoreConfig`
- Them block `quality_score` trong `brain_full.yaml`.
- Mo rong BRAIN result records va simulation results voi check summary fields.
- Adapter va service parse raw BRAIN payload thanh check summary va submission eligibility.

Tac dong hanh vi:

- Quality score gan hon voi ket qua check that cua BRAIN.
- Near-miss voi robustness warnings khong bi xem nhu fail nang.
- Structural-risk checks duoc tai su dung trong search-space filter, recipe parent selection va quality polish parent selection.

### 4. Persistence va Schema

File chinh:

- `storage/sqlite.py`
- `storage/brain_result_store.py`
- `storage/repositories/alpha_repository.py`
- `storage/repositories/recipe_repository.py`
- `domain/brain.py`
- `domain/simulation.py`

Noi dung thay doi:

- Them column vao `brain_results`:
  - `check_summary_json`
  - `hard_fail_checks_json`
  - `warning_checks_json`
  - `blocking_warning_checks_json`
  - `derived_submit_ready`
- Them required-column migration entries cho SQLite DB cu.
- Them table `field_diagnostics` va index.
- Cap nhat insert/upsert/load logic trong result store.
- Cap nhat alpha/recipe repository query de expose check fields cho generation logic.

Tac dong hanh vi:

- DB cu se duoc upgrade qua duong repository initialization.
- Stored BRAIN result history co du thong tin hon de tinh check-aware scoring va priors.

### 5. Operator Diversity Boost

File chinh:

- `generator/operator_diversity.py` moi
- `generator/engine.py`
- `generator/genome_builder.py`
- `generator/guided_generator.py`
- `config/models/adaptive.py`
- `config/brain_full.yaml`
- `tests/test_generator_engine.py`

Noi dung thay doi:

- Them `OperatorDiversityBoostConfig` va defaults:
  - dominant operators
  - underused operators
  - expected report date fields
  - seed correlation pairs
  - decay/boost params
- Them `OperatorDiversityState` de tinh adjusted operator weights va metrics.
- `AlphaGenerationEngine` co the tao targeted expression cho:
  - `ts_corr`
  - `ts_covariance`
  - `days_from_last_change`
- Them rendering targeted expression voi seed pairs, random diverse pairs, report-date fields va normalization wrappers.
- `GenomeBuilder` nhan operator diversity state de adjust wrapper, smoothing, primitive, pair va conditioning operator choices.
- `GuidedGenerator` pass operator diversity state qua exploit/explore generation.
- Generation stats co operator-diversity metrics.

Tac dong hanh vi:

- Generator giam overuse cac operator dominant.
- Tang co hoi sample cac operator underused nhung van nam trong allowlist.
- Search-space operator priors co the feed vao operator weights cua generator.

### 6. Recipe-Guided va Group-Relative Generation

File chinh:

- `services/recipe_guided_generator.py`
- `services/brain_batch_service.py`
- `storage/repositories/recipe_repository.py`
- `tests/test_recipe_guided_generator.py`

Noi dung thay doi:

- Them source `group_relative` voi budget cap.
- Them cac group-relative recipe groups:
  - A: relative fundamental group transforms
  - B: momentum/returns group neutralization
  - C: earnings short-window/group transforms
  - D: liquidity/volume group transforms
- Group keys lay tu field registry, uu tien `subindustry` va `sector`, fallback sang `industry` va `country`.
- Weighted group-key cycle uu tien subindustry.
- Them primary-field cap de tranh mot field dominate group-relative drafts.
- Them stats theo source/group.
- Recipe parent selection bo qua structural-risk blockers.
- Brain batch metrics tach ro generation source counts, gom ca `group_relative`.

Tac dong hanh vi:

- Recipe-guided generation sinh nhieu cau truc cross-sectional/group-relative hon.
- Batch metrics cho thay ro source nao generate/selected candidates.

### 7. CLI Diagnostics va Backfill

File chinh:

- `cli/app.py`
- `cli/commands/backfill_brain_checks.py` moi
- `cli/commands/diagnose_fields.py` moi
- `tests/test_backfill_brain_checks.py`
- `tests/test_field_diagnostics.py`

Noi dung thay doi:

- Them command `backfill-brain-checks`:
  - scan stored BRAIN results
  - parse raw check payloads
  - persist check summary columns
  - recompute quality scores
  - idempotent
- Them command `diagnose-fields`:
  - dry-run mac dinh
  - tao diagnostic expressions cho raw value, non-zero coverage, update frequency va absolute bounds
  - chi submit BRAIN khi co `--submit`
  - persist diagnostic rows vao `field_diagnostics`

Tac dong hanh vi:

- Co the upgrade DB history cu ma khong can rerun simulations.
- Co the tao field-level BRAIN diagnostics de dung cho search-space penalties.

### 8. Service Status va Runtime Visibility

File chinh:

- `services/status_service.py`
- `cli/commands/service_status.py`
- `tests/test_service_status_command.py`

Noi dung thay doi:

- Service status snapshot them:
  - derived submit-ready counts
  - top hard fail checks
  - top blocking warning checks
- Human output hien readiness/check details cho recent results.

Tac dong hanh vi:

- De chan doan hon khi alpha fail do BRAIN checks nao.

### 9. Config Changes

File chinh:

- `config/brain_full.yaml`

Thay doi quan trong:

- Enable `operator_diversity_boost`.
- Them dominant/underused operators, seed pairs va report-date fields cho operator diversity.
- Lam winner prior it aggressive hon va co statistical guard.
- Them field/operator floor va exploration budget.
- Mo rong lane operator allowlist voi cac operator nhu `ts_delta`, `ts_decay_linear`, `ts_rank`, `group_rank`, `group_zscore`, `group_neutralize`.
- Chinh fresh budget:
  - `max_fresh_budget_fraction`: 0.26 -> 0.30
  - `fresh_spillover_fraction`: 0.03 -> 0.20
- Tighten quality polish parent thresholds:
  - `min_parent_fitness`: 0.02 -> 0.20
  - `min_parent_sharpe`: 0.03 -> 0.50
  - `min_parent_turnover`: them 0.01
  - `max_parent_turnover`: 1.00 -> 0.60
- Them structural quality polish transforms va budget percentages.
- Them block `quality_score`.
- Tang BRAIN timeout:
  - `timeout_seconds`: 600 -> 1000

Tac dong hanh vi:

- Profile chon parent cho quality-polish khat khe hon.
- Search space exploratory va da dang ve structure hon.
- Runtime doi BRAIN simulations lau hon truoc khi timeout.

## File Untracked

Source/test/docs dang untracked:

- `cli/commands/backfill_brain_checks.py`
- `cli/commands/diagnose_fields.py`
- `config/models/quality.py`
- `core/brain_checks.py`
- `generator/operator_diversity.py`
- `tests/test_backfill_brain_checks.py`
- `tests/test_brain_checks.py`
- `tests/test_field_diagnostics.py`
- `tests/test_generator_engine.py`
- `docs/code_quality_report_for_claude.md`
- `docs/current_brain_search_space_for_claude.md`
- `docs/recent_changes_report_2026-05-03.md`

Artifact dang untracked:

- `dist/wq_tool_code_review_2026-05-01.zip`
- `dist/wq_tool_code_review_20260501_183440.zip`
- `dist/wq_tool_code_upload_2026-04-26.zip`
- `dist/wq_tool_source_20260426_195957.zip`

Khuyen nghi:

- Khong commit `dist/*.zip` neu day chi la generated archive.
- Nen them hoac xac nhan rule `.gitignore` cho generated archives.

## Verification

Lenh da chay:

```powershell
git status --short
git diff --stat
git diff --name-only
git diff --check
py -3.12 -m pytest tests/test_quality_polisher.py tests/test_search_space_filter.py tests/test_config.py tests/test_quality_score.py tests/test_recipe_guided_generator.py tests/test_brain_checks.py tests/test_backfill_brain_checks.py tests/test_field_diagnostics.py tests/test_generator_engine.py tests/test_brain_integration.py tests/test_brain_api_auth.py tests/test_service_status_command.py -q
```

Ket qua:

- `git diff --check`: khong co whitespace error can block; chi co canh bao line ending.
- Pytest: 142 passed.

## Rui ro va ghi chu review

1. Diff lon va cham nhieu lop: generation, scoring, storage, service runtime, CLI va config. Chi nen commit chung neu deployment plan muon dua tat ca schema/config change len cung luc. Neu khong, nen split commit:
   - brain checks va quality score
   - search-space prior changes
   - structural quality polish
   - operator diversity
   - group-relative recipes
   - CLI diagnostics/backfill

2. Winner-prior cache moi thread-safe trong mot Python process, nhung khong dong bo giua nhieu service process. No khong giai quyet van de duplicate service process gay SQLite lock.

3. Code hien tai ky vong `brain_results` co cac column moi. SQLite required-column migration da cover, nhung khi deploy can dam bao service khoi dong qua normal repository initialization truoc khi doc cac field moi.

4. `field_diagnostics` la table moi. Script backup/inspection ngoai he thong neu assume fixed schema co the can update.

5. `brain_full.yaml` thay doi hanh vi kha manh. Dac biet thresholds quality-polish parent chat hon nhieu va BRAIN timeout dai hon.

6. `dist/` co cac zip artifact lon dang untracked. Kha nang cao nen de ngoai git.

7. Git bao LF-to-CRLF warning tren nhieu file. Day khong lam fail test, nhung co the tao noisy diff sau nay neu line-ending policy khong ro.

## Ket luan

Cac thay doi gan day dua adaptive search sang huong structural va check-aware hon:

- quality polish tao bien the thay doi cau truc thay vi chi doi lookback
- winner prior bot overfit vao sample nho
- BRAIN checks duoc parse va dung lai trong scoring/filtering
- generator co co che tang operator diversity
- recipe-guided co them group-relative structures
- CLI co cong cu backfill va field diagnostics

Rui ro chinh hien tai khong nam o test failure. Rui ro nam o hygiene khi deploy: diff qua lon, schema moi, line-ending churn, artifact zip untracked, va van de multi-process SQLite runtime.
