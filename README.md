# WQ Tool

`WQ Tool` la mot alpha research framework theo huong WorldQuant/BRAIN.
He thong nay hien co 2 lop ro rang:

- local research layer: field registry, operator registry, structured generation, validation, mutation, memory, storage
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
- sinh alpha theo template co cau truc, type-safe, co memory
- validate expression truoc khi evaluate/simulate
- persist lineage, pattern memory, field scores, va metadata traceability
- submit candidate vao BRAIN qua `manual` hoac `api` adapter
- normalize ket qua BRAIN, luu rejection reason, submission eligibility, raw payload
- chay `generate -> simulate -> learn -> mutate -> repeat`
- chay `run-service` 24/7 de lien tuc poll pending jobs va tao batch moi an toan

## Cau truc muc cao

```text
adapters/    SimulationAdapter + BRAIN manual/api backends
alpha/       Parser / AST / validator / evaluator
cli/         Argparse wiring
data/        Data loader + field registry
features/    Operator registry + transforms
generator/   Template generator + mutation + guided generation
memory/      Pattern memory + structural signatures
services/    Brain service / closed loop / candidate selection / local services
storage/     SQLite schema + repository + result stores
workflows/   Thin workflow wrappers cho CLI
docs/        Architecture / pipeline / config / development notes
```

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

1. generate candidate co cau truc
2. validate/prefilter local
3. submit top-N vao BRAIN
4. thu ket qua that tu BRAIN
5. luu metrics + rejection reasons + lineage
6. cap nhat memory
7. mutate candidate manh
8. lap lai theo so round

### Service mode 24/7

```bash
python main.py run-service --config config/dev.yaml
```

`run-service` la foreground process de chay duoi Task Scheduler, NSSM, systemd, hoac supervisor tuong tu.

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
- in URL ra terminal va gui mail neu SMTP da cau hinh
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
- gui link Persona qua mail neu SMTP da cau hinh
- tu dong polling Persona trong che do headless
- local session-cookie cache de tai su dung giua cac command

Khuyen nghi:

1. dat `brain.backend: api`
2. chay `python main.py brain-login --config <config>.yaml`
3. nhap email/password trong terminal, hoac luu trong `secrets/brain_credentials.json`
4. neu BRAIN yeu cau Persona, mo URL va quet mat
5. neu da cau hinh SMTP, tool gui link Persona qua mail va tu dong doi ban xac thuc
6. session cookie se duoc luu vao `brain.session_path`

Luu y bao mat:

- tool khong luu password
- neu ban muon chay 24/7, hay luu BRAIN credential + SMTP app password trong `secrets/brain_credentials.json`
- tool co the luu session cookie cuc bo de tai su dung
- mac dinh file session nam trong `outputs/`, da duoc `.gitignore`

## Storage va traceability

Ngoai cac bang local cu, repo da co them:

- `submission_batches`
- `submissions`
- `brain_results`
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
