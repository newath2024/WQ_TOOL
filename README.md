# WQ Tool

`WQ Tool` la mot alpha research framework theo huong WorldQuant/BRAIN.
He thong nay hien co 3 lop ro rang:

- local research layer: field registry, operator registry, genome-based generation, validation, mutation, memory, storage
- BRAIN integration layer: submit alpha vao BRAIN, lay ket qua that, hoc tu outcome that, va chay closed-loop
- service layer: giu service loop 24/7, resume job dang chay, heartbeat, lock, notification, va safe shutdown

Quan trong:

- local logic KHONG phai la backtest thay the BRAIN
- BRAIN simulation la source of truth cho closed-loop workflow
- manual backend ton tai de workflow van dung duoc truoc khi API that san sang
- API backend la scaffold san sang tich hop, nhung khong bịa endpoint

## Kha nang hien tai

- nap OHLCV local + metadata group/factor/mask
- nap field catalog tu snapshot BRAIN va field values tu long CSV
- sinh alpha tu `Genome -> AST -> expression`, co motif grammar, wrapper stack, va metadata traceability
- enrich operator registry tu BRAIN operator catalog de dung `summary/details` cho mutation, repair, va motif selection
- validate expression truoc khi evaluate/simulate
- mutate bang 5 che do: `exploit_local`, `structural`, `crossover`, `novelty`, `repair`
- chon candidate bang multi-objective + diversity-preserving selection
- persist lineage, pattern memory, rich case memory, field scores, va metadata traceability
- hoc theo kien truc hierarchical: local memory theo `region + compatible regime`, kem global priors theo profile khong gom region
- submit candidate vao BRAIN qua `manual` hoac `api` adapter
- normalize ket qua BRAIN, luu rejection reason, submission eligibility, raw payload
- chay `generate -> simulate -> learn -> mutate -> repeat`
- chay `run-service` 24/7 de lien tuc poll pending jobs va tao batch moi an toan

## Genome Evolution Engine

Generator mac dinh khong con dua tren template string phang. Moi alpha moi duoc bieu dien boi genome gom:

- feature genes
- transform/motif genes
- horizon/lookback genes
- wrapper genes
- regime/conditioning genes
- turnover-control genes
- complexity genes

Genome duoc render qua AST roi moi normalize thanh expression string de giu nguyen external contract voi validator, evaluator, CLI, va BRAIN adapter.

Motif grammar ban dau gom:

- momentum
- mean reversion
- volatility-adjusted momentum
- spread
- ratio
- residualized signal
- regime-conditioned signal
- group-relative signal
- liquidity-conditioned signal

Bo may search moi cung them:

- gene-level crossover giua parent manh
- novelty search dua tren structural distance
- repair policy de sua candidate loi truoc khi loai
- case memory forward-only trong bang `alpha_cases`
- region-aware pattern memory + case memory voi global fallback co cau hinh
- hard diversity caps theo family, field category, horizon, va operator path

## Region-aware learning

Closed-loop learning khong con dung mot memory pool global share-toan-bo.
He thong moi tach thanh 2 lop:

- local memory theo `region + compatible regime` de parent pool, mutation stats, fail tags, template priors, diversity, turnover/complexity behavior khong bi cross-region contamination
- global priors theo profile khong gom region de cold-start region moi hoac region co sample count thap

Nguyen tac:

- parser / validator / normalization / operator registry / grammar / orchestration van share chung
- top parent history va parent selection mac dinh chi dung local region
- scoring, motif/template priors, mutation mode priors, va duplicate-family diagnostics co the blend local + global
- blend la explicit va configurable qua `adaptive_generation.region_learning`

## Cau truc muc cao

```text
adapters/    SimulationAdapter + BRAIN manual/api backends
alpha/       Parser / AST / validator / evaluator
cli/         Argparse wiring
data/        Data loader + field registry
features/    Operator registry + transforms
generator/   Genome + motif grammar + mutation + crossover + novelty + repair
memory/      Pattern memory + case memory + structural signatures
services/    Brain service / closed loop / multi-objective selection / diversity
storage/     SQLite schema + repository + result stores
workflows/   Thin workflow wrappers cho CLI
docs/        Architecture / pipeline / config / development notes
```

Knowledge-playbook cho alpha research:

- `docs/finding_alpha/README.md`: quick lookup tu van de thuc te (`turnover`, `correlation`, `overfitting`, `automated search`, `fundamental recipes`) sang notes paraphrase tu `FindingAlpha.pdf`

## Workflow local va workflow BRAIN

### Local workflow cu

Van duoc giu nguyen:

```bash
python main.py generate --config config/dev.yaml
python main.py evaluate --config config/dev.yaml
python main.py run-full-pipeline --config config/dev.yaml
```

Workflow nay dung local evaluation/backtest de screening va debug.

### BRAIN-first workflow moi

```bash
python main.py sync-field-catalog --config config/dev.yaml
python main.py brain-login --config config/dev.yaml
python main.py export-brain-candidates --config config/dev.yaml
python main.py import-brain-results --config config/dev.yaml --path outputs/brain_manual/manual_results.csv
python main.py run-brain-sim --config config/dev.yaml
python main.py run-closed-loop --config config/dev.yaml
python main.py run-service --config config/dev.yaml
```

Closed-loop BRAIN:

1. build genome va render expression
2. validate/prefilter local
3. multi-objective selection + diversity-preserving filtering
4. submit top-N vao BRAIN
5. thu ket qua that tu BRAIN
6. luu metrics + rejection reasons + lineage
7. cap nhat pattern memory + case memory
8. mutate/crossover/repair candidate manh
9. lap lai theo so round

### Service mode 24/7

```bash
python main.py run-service --config config/dev.yaml
python main.py service-status --config config/dev.yaml
```

`run-service` la foreground process de chay duoi Task Scheduler, NSSM, systemd, hoac supervisor tuong tu.
`service-status` la lenh nhanh de xem service dang o run nao, batch nao, con bao nhieu alpha pending, va ket qua gan nhat.

Google Drive snapshot backup:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\backup_to_google_drive.ps1 -GoogleDriveRoot "C:\Users\<you>\Google Drive"
powershell -ExecutionPolicy Bypass -File .\tools\register_google_drive_backup_task.ps1 -GoogleDriveRoot "C:\Users\<you>\Google Drive"
```

`backup_to_google_drive.ps1` tao snapshot an toan cua `*.sqlite3`, `outputs/`, `progress_logs/`, `config/`, va mot so file repo can thiet vao `Google Drive\WQ_TOOL_backups`.
Script mac dinh khong copy `outputs/brain_api_session.json` de tranh sync session API.

Moi service tick:

1. renew/acquire DB lease lock
2. kiem tra session/auth
3. resume va poll pending jobs neu co
4. persist ket qua + cap nhat memory khi batch hoan tat
5. neu khong con pending jobs thi tao batch moi va submit len BRAIN
6. ghi heartbeat/counters vao `service_runtime`
7. sleep theo service scheduler

Resume sau restart:

- service doc `service_runtime` + `submissions` + `submission_batches`
- neu con job `submitted/running` thi chi poll tiep, khong submit batch moi
- neu batch dang `submitting` ma DB khong du `job_id` de biet da submit toi dau, batch bi `paused_quarantine`

Lock behavior:

- chi cho phep 1 service instance tren moi DB/profile
- DB lease lock tu dong het han neu process chet
- instance khac co the takeover sau khi lease cu het han

Persona behavior:

- service mode dung non-interactive auth
- neu BRAIN doi Persona, service pause viec submit batch moi
- neu da cau hinh Telegram bot va `service.persona_confirmation_required: true`, service se hoi ban co san sang xac thuc khong truoc khi xin link moi
- in URL ra terminal va gui Telegram neu bot da cau hinh
- retry auth cham theo `service.persona_retry_interval_seconds`, khong spam lien tuc

## Manual backend

Mac dinh `brain.backend = manual`.

Luong manual:

1. `export-brain-candidates` tao CSV trong `outputs/brain_manual/`
2. submit thu cong len BRAIN
3. luu ket qua thu cong vao CSV
4. `import-brain-results --path <file.csv>`
5. chay tiep `run-closed-loop` hoac xem traceability trong DB

Export CSV co:

- `candidate_id`
- `job_id`
- `expression`
- `template_name`
- `fields_used`
- `operators_used`
- `generation_metadata_json`
- `sim_config_json`

## API backend

`BrainApiAdapter` da co:

- auth/session wrapper
- payload builder
- response parser
- retry hook
- rate-limit hook
- status/result polling hooks
- interactive login bang email/password
- support Persona/face-scan khi BRAIN yeu cau
- doc credential tu `secrets/brain_credentials.json`
- gui link Persona qua Telegram bot neu da cau hinh
- tu dong polling Persona trong che do headless
- local session-cookie cache de tai su dung giua cac command

Khuyen nghi:

1. dat `brain.backend: api`
2. chay `python main.py brain-login --config <config>.yaml`
3. nhap email/password trong terminal, hoac luu trong `secrets/brain_credentials.json`
4. neu BRAIN yeu cau Persona, mo URL va quet mat
5. neu da cau hinh Telegram bot, tool gui link Persona qua Telegram
6. session cookie se duoc luu vao `brain.session_path`

Luu y bao mat:

- tool khong luu password
- neu ban muon chay 24/7, hay luu BRAIN credential + Telegram bot token trong `secrets/brain_credentials.json`
- tool co the luu session cookie cuc bo de tai su dung
- mac dinh file session nam trong `outputs/`, da duoc `.gitignore`

## Storage va traceability

Ngoai cac bang local cu, repo da co them:

- `submission_batches`
- `submissions`
- `brain_results`
- `alpha_cases`
- `manual_imports`
- `closed_loop_runs`
- `closed_loop_rounds`
- `service_runtime`

Moi candidate co the truy vet duoc:

- run nao tao ra
- round nao submit len BRAIN
- batch/job nao tuong ung
- config simulation nao da dung
- ket qua metrics nao tra ve
- co rejection khong
- ly do reject la gi
- co duoc chon de mutate tiep khong

## Config moi

`brain` block tach rieng khoi `generation`, `simulation`, `loop`.
`adaptive_generation` gio co them config cho exploration/exploitation, novelty, mutation modes, crossover, diversity caps, va repair policy.

Vi du:

```yaml
brain:
  backend: api
  region: USA
  universe: TOP3000
  delay: 1
  neutralization: sector
  decay: 0
  truncation: 0.08
  pasteurization: true
  unit_handling: verify
  nan_handling: off
  session_path: outputs/brain_api_session.json

loop:
  rounds: 5
  generation_batch_size: 100
  simulation_batch_size: 20
  poll_interval_seconds: 10
  timeout_seconds: 600
  mutate_top_k: 10
  max_children_per_parent: 5

service:
  enabled: false
  tick_interval_seconds: 5
  idle_sleep_seconds: 30
  poll_interval_seconds: 10
  max_pending_jobs: 20
  cooldown_seconds: 300
  lock_name: brain-service
```

Region-aware learning:

```yaml
adaptive_generation:
  region_learning:
    enabled: true
    local_scope: region_regime
    global_prior_scope: match_non_region_regime
    blend_mode: linear_ramp
    min_local_pattern_samples: 20
    full_local_pattern_samples: 100
    min_local_case_samples: 10
    full_local_case_samples: 50
    allow_global_parent_fallback: false
```

Memory inspection co them 3 scope:

```bash
python main.py memory-top-patterns --config config/dev.yaml --scope local
python main.py memory-top-patterns --config config/dev.yaml --scope global
python main.py memory-top-patterns --config config/dev.yaml --scope blended
```

Config cu van load duoc. Command local cu khong doi nghia.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Test

```bash
pytest -q
```

## Gioi han hien tai

- manual backend can buoc import ket qua thu cong truoc khi closed-loop tiep tuc hoc day du
- API backend moi la scaffold va chua duoc bind vao endpoint BRAIN that
- local evaluation van ton tai cho workflow cu, nhung khong duoc xem la truth trong BRAIN closed-loop
- `run-service` v1 la single-machine foreground service, chua huong toi distributed coordination
- batch `paused_quarantine` can operator xem va xu ly thu cong truoc khi service tiep tuc submit batch moi

## Tai lieu

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Pipeline](docs/pipeline.md)
- [Development](docs/development.md)
