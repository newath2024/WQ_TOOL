# Configuration

## Tong quan

Config hien tai tach thanh 4 nhom ro rang:

- `generation`: local alpha search space
- `simulation`: local evaluation/backtest cu
- `brain`: external BRAIN simulation settings
- `loop`: closed-loop orchestration settings

Workflow local cu van doc config cu binh thuong.
Workflow BRAIN moi doc them `brain` va `loop`.

## Generation

```yaml
generation:
  allowed_fields: ["open", "high", "low", "close", "volume", "returns"]
  allowed_operators: ["rank", "delta", "ts_mean", "ts_std", "group_rank"]
  lookbacks: [2, 3, 5, 10]
  max_depth: 4
  complexity_limit: 20
  template_count: 40
  grammar_count: 40
  mutation_count: 20
  normalization_wrappers: ["rank", "zscore", "sign"]
  random_seed: 7
  field_catalog_paths: []
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
- generator mac dinh hien tai la template-driven, khong grammar-random-first

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
- `generation_batch_size`: so candidate generate moi round
- `simulation_batch_size`: top-N gui BRAIN
- `mutate_top_k`: so parent lay tu BRAIN result
- `max_children_per_parent`: budget mutation cho round sau

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
- `brain.delay` va `brain.max_retries` phai hop le

Neu `brain` hoac `loop` khong co trong YAML:

- config se duoc fill bang default an toan
- workflow local cu van khong bi anh huong
