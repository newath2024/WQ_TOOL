# Configuration

## Tong quan

Config hien tai tach thanh 4 nhom ro rang:

- `generation`: local alpha search space
- `adaptive_generation`: genome evolution policy, memory-guided mutation, novelty, diversity, repair
- `simulation`: local evaluation/backtest cu
- `brain`: external BRAIN simulation settings
- `loop`: closed-loop orchestration settings
- `service`: foreground 24/7 service mode settings

Workflow local cu van doc config cu binh thuong.
Workflow BRAIN moi doc them `brain` va `loop`.

## Generation

```yaml
generation:
  allowed_fields: ["open", "high", "low", "close", "volume", "returns"]
  allowed_operators: ["rank", "ts_delta", "ts_mean", "ts_std_dev", "group_rank"]
  lookbacks: [2, 3, 5, 10]
  max_depth: 4
  complexity_limit: 20
  template_count: 40
  grammar_count: 40
  mutation_count: 20
  normalization_wrappers: ["rank", "zscore", "sign"]
  random_seed: 7
  field_catalog_paths: []
  operator_catalog_paths:
    - inputs/wq_snapshots/2026-03-29/operators/brain_operator_catalog.json
  field_value_paths: []
  field_score_weights:
    coverage: 0.50
    usage: 0.30
    category: 0.20
  category_weights:
    price: 1.00
    volume: 0.85
    fundamental: 0.95
    analyst: 0.90
    model: 0.85
    sentiment: 0.75
    risk: 0.70
    macro: 0.65
    group: 0.60
    other: 0.50
  template_weights: {}
  template_pool_size: 200
  max_turnover_bias: 0.35
```

Ghi chu:

- `grammar_count` van duoc chap nhan de backward compatibility
- generator mac dinh hien tai la genome-based. Expression string van la external contract, nhung duoc render tu `Genome -> AST -> normalized expression`
- `operator_catalog_paths` cho phep load operator catalog da export tu BRAIN de enrich registry bang `summary/details/tags/constraints`
- config/sample moi uu tien ten operator chuan cua BRAIN nhu `ts_delta`, `ts_corr`, `ts_covariance`, `ts_decay_linear`, `ts_std_dev`
- alias local cu nhu `delta`, `correlation`, `covariance`, `decay_linear`, `ts_std` van duoc registry chap nhan de backward compatibility, nhung khong con la mac dinh de submit BRAIN

## Adaptive generation

```yaml
adaptive_generation:
  enabled: true
  memory_scope: regime
  success_rule: validation_first
  strategy_mix:
    guided_mutation: 0.40
    memory_templates: 0.30
    random_exploration: 0.20
    novelty_behavior: 0.10
  exploration_epsilon: 0.10
  sampling_temperature: 0.75
  family_cap_fraction: 0.25
  parent_pool_size: 30
  novelty_reference_top_k: 20
  min_pattern_support: 3
  pattern_decay: 0.98
  exploration_ratio: 0.35
  novelty_weight: 0.25
  mutation_mode_weights:
    exploit_local: 0.35
    structural: 0.25
    crossover: 0.15
    novelty: 0.15
    repair: 0.10
  crossover_rate: 0.15
  diversity:
    max_family_fraction: 0.25
    max_field_category_fraction: 0.50
    max_horizon_bucket_fraction: 0.40
    max_operator_path_fraction: 0.40
    exploration_quota_fraction: 0.20
    min_structural_distance: 0.08
  repair_policy:
    enabled: true
    max_attempts: 3
    allow_complexity_reduction: true
    allow_turnover_reduction: true
    allow_wrapper_cleanup: true
    allow_group_fixups: true
```

Y nghia:

- `exploration_ratio`: ti le quota danh cho genome moi/novel candidates truoc khi mutation-heavy pool chiem het budget
- `novelty_weight`: trong so novelty trong objective prediction va ranking
- `mutation_mode_weights`: xac suat co ban cho 5 mutation modes. He thong van co the dieu chinh theo case memory va failure tags
- `crossover_rate`: gioi han tan suat child lai tu 2 parent thay vi mutation 1 parent
- `diversity.max_family_fraction`: tran ti le candidate cung family signature trong top set
- `diversity.max_field_category_fraction`: tran ti le candidate tap trung vao cung field category
- `diversity.max_horizon_bucket_fraction`: tran ti le candidate tap trung vao cung bucket horizon/lookback
- `diversity.max_operator_path_fraction`: tran ti le candidate co operator path qua giong nhau
- `diversity.exploration_quota_fraction`: quota rieng cho exploration/novelty ma exploit candidate khong duoc an het
- `diversity.min_structural_distance`: khoang cach toi thieu de 2 genome khong bi xem la near-clone
- `repair_policy.enabled`: bat/tat bounded repair pass sau render/mutation
- `repair_policy.max_attempts`: so lan sua toi da truoc khi discard candidate
- `repair_policy.allow_complexity_reduction`: cho phep tu dong giam complexity neu gene vuot budget
- `repair_policy.allow_turnover_reduction`: cho phep thay gene/wrapper de ha turnover pressure
- `repair_policy.allow_wrapper_cleanup`: cho phep cat wrapper lap/thua
- `repair_policy.allow_group_fixups`: cho phep sua group/operator combination sai

Case memory moi se persist vao bang `alpha_cases` va duoc dung de du doan objective cho fresh candidates, chon mutation mode, va tron exploit/explore theo regime.

## Brain config

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
  poll_interval_seconds: 10
  timeout_seconds: 600
  max_retries: 3
  batch_size: 20
  manual_export_dir: outputs/brain_manual
  api_base_url: ""
  api_auth_env: BRAIN_API_TOKEN
  email_env: BRAIN_API_EMAIL
  password_env: BRAIN_API_PASSWORD
  credentials_file: secrets/brain_credentials.json
  session_path: outputs/brain_api_session.json
  auth_expiry_seconds: 14400
  open_browser_for_persona: true
  persona_poll_interval_seconds: 15
  persona_timeout_seconds: 1800
  rate_limit_per_minute: 60
```

Y nghia:

- `backend`: `manual` hoac `api`
- `manual_export_dir`: noi ghi CSV de submit thu cong
- `api_base_url`/`api_auth_env`: de san cho backend API
- `email_env`/`password_env`: bien moi truong tuy chon; neu khong co, tool se prompt trong terminal
- `credentials_file`: file JSON local de luu `brain.email`, `brain.password`, va SMTP config cho Persona mail
- `session_path`: file luu session cookie sau khi login thanh cong
- `auth_expiry_seconds`: xin session toi da 14400 giay theo tai lieu BRAIN
- `open_browser_for_persona`: tu mo URL neu BRAIN yeu cau quet mat
- `persona_poll_interval_seconds`: tan suat polling link Persona trong che do headless
- `persona_timeout_seconds`: thoi gian cho toi da de doi ban quet mat

## File credentials local

Tool mac dinh doc file `secrets/brain_credentials.json`; mau tham khao nam o `secrets/brain_credentials.example.json`.

File nay da duoc them vao `.gitignore`, nen ban co the luu credential local ma khong bi commit.

Mau JSON:

```json
{
  "brain": {
    "email": "your-brain-email@example.com",
    "password": "your-brain-password"
  },
  "persona_notification": {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_username": "your-mail@gmail.com",
    "smtp_password": "your-app-password",
    "from_email": "your-mail@gmail.com",
    "to_email": "your-mail@gmail.com",
    "use_tls": true
  }
}
```

Neu ban muon chay 24/7:

1. dien `brain.email` va `brain.password`
2. dien SMTP credential de gui mail
3. khi BRAIN tra ve `persona`, tool se gui link qua mail va tu dong polling cho den khi ban quet mat xong

## Dang nhap API tu terminal

Neu `brain.backend: api`, co 2 cach:

### Cach 1: prompt tu tool

```bash
python main.py brain-login --config config/dev.yaml
```

Tool se:

1. prompt email
2. prompt password bang `getpass`
3. neu BRAIN tra ve `persona`, in URL va co the mo browser
4. neu `credentials_file` co SMTP config, gui link Persona qua mail
5. tu dong polling cho den khi ban quet mat xong hoac het timeout
6. luu session cookie vao `brain.session_path`

### Cach 2: dung bien moi truong

```powershell
$env:BRAIN_API_EMAIL="you@example.com"
$env:BRAIN_API_PASSWORD="your-password"
python main.py brain-login --config config/dev.yaml
```

Password khong duoc luu vao file config.

## Loop config

```yaml
loop:
  rounds: 5
  generation_batch_size: 100
  simulation_batch_size: 20
  poll_interval_seconds: 10
  timeout_seconds: 600
  mutate_top_k: 10
  max_children_per_parent: 5
  rejection_filters: []
  archive_thresholds: {}
```

Y nghia:

- `rounds`: so vong closed-loop
- `generation_batch_size`: so candidate generate moi round, bao gom fresh genome, novelty candidate, va mutation/crossover children
- `simulation_batch_size`: top-N gui BRAIN
- `mutate_top_k`: so parent lay tu BRAIN result
- `max_children_per_parent`: budget mutation cho round sau

## Service config

```yaml
service:
  enabled: false
  tick_interval_seconds: 5
  idle_sleep_seconds: 30
  poll_interval_seconds: 10
  max_pending_jobs: 20
  max_consecutive_failures: 5
  cooldown_seconds: 300
  heartbeat_interval_seconds: 30
  lock_name: brain-service
  lock_lease_seconds: 60
  resume_incomplete_jobs: true
  shutdown_grace_period_seconds: 30
  stuck_job_after_seconds: 1800
  persona_retry_interval_seconds: 300
  persona_email_cooldown_seconds: 900
```

Y nghia:

- `tick_interval_seconds`: nhac scheduler khi khong co state dac biet
- `idle_sleep_seconds`: sleep khi khong co pending jobs va khong tao duoc batch moi
- `poll_interval_seconds`: tan suat poll pending jobs trong service mode
- `max_pending_jobs`: gioi han pending jobs cho moi service instance
- `max_consecutive_failures`: qua nguong nay service vao cooldown
- `cooldown_seconds`: thoi gian tam dung sau khi loi lap lai
- `heartbeat_interval_seconds`: tan suat toi da de cap nhat heartbeat khi dang pause/cooldown
- `lock_name`: ten DB lease lock; moi DB/profile chi nen co 1 service dung ten nay
- `lock_lease_seconds`: thoi gian lease song truoc khi instance khac duoc takeover
- `resume_incomplete_jobs`: boot lai se resume `service_run_id` cu neu co
- `shutdown_grace_period_seconds`: cua so de service dung an toan duoi supervisor
- `stuck_job_after_seconds`: danh dau job bi treo de surface trong state/log
- `persona_retry_interval_seconds`: nhan lai auth sau khi cho Persona
- `persona_email_cooldown_seconds`: throttle email Persona

## Field catalog va runtime field values

### Catalog metadata

Co the nap tu JSON/CSV snapshot BRAIN:

- `field_catalog_paths`

Field score:

```text
field_score =
  0.5 * coverage_norm +
  0.3 * usage_norm +
  0.2 * category_weight
```

### Runtime field values

Long CSV:

```csv
timestamp,symbol,field,value,dataset,field_type,category
2021-01-01,AAA,pe_ratio,10.5,fundamental,matrix,fundamental
2021-01-01,AAA,sector_label,technology,meta,vector,group
```

Bat buoc:

- `timestamp`
- `symbol`
- `field`
- `value`

Tuy chon:

- `dataset`
- `field_type`
- `category`
- `timeframe`
- `description`
- `subcategory`
- `region`
- `universe`
- `delay`

## Validation rules tren config

Loader hien tai se validate:

- `brain.backend` phai la `manual` hoac `api`
- cac timeout/poll/batch size/round count phai > 0
- cac tham so `service.*_seconds`, lease lock, cooldown, va pending cap phai > 0
- `brain.delay` va `brain.max_retries` phai hop le

Neu `brain` hoac `loop` khong co trong YAML:

- config se duoc fill bang default an toan
- workflow local cu van khong bi anh huong
- `adaptive_generation` moi se duoc fill bang default de giu backward compatibility voi YAML cu
