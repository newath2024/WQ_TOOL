# Pipeline

> Normative note: lifecycle, orchestration, timeout, BRAIN recovery, and observability behavior is governed by `docs/alpha_pipeline_spec.md`. This file is explanatory context, not the behavioral source of truth.

## 1. Local research pipeline

Luong local cu van ton tai:

1. load data
2. generate candidate
3. validate
4. evaluate/backtest local
5. filter + dedup + rank
6. update memory local

Workflow nay huu ich cho:

- debug
- smoke testing
- screening tren sample/local dataset

Nhung no KHONG thay the BRAIN simulation trong closed-loop moi.

## 2. BRAIN-first single batch

`run-brain-sim` va `export-brain-candidates` di theo luong:

1. load dataset + field registry
2. generate structured candidates
3. local validate va dedup
4. pre-rank theo heuristic + diversity + memory
5. submit top-N len BRAIN
6. poll/import ket qua
7. normalize va persist

Neu backend la `api` va chua co session:

1. chay `brain-login`, hoac
2. de `run-brain-sim` / `run-closed-loop` tu doc `secrets/brain_credentials.json`
3. neu BRAIN yeu cau Persona, tool se dua URL de ban quet mat
4. neu credentials file co Telegram bot config, tool se gui link Persona qua Telegram
5. tool tu dong polling cho den khi ban quet mat xong
6. sau khi xac thuc xong, session cookie duoc tai su dung cho cac command sau

`export-brain-candidates`:

- force workflow manual export
- khong cho result that
- dung khi can submit thu cong

`run-brain-sim`:

- dung backend trong `brain.backend`
- voi `manual`, batch se dung o `manual_pending`
- voi `api`, service se poll va thu result neu backend da cau hinh

## 3. Closed-loop pipeline

`run-closed-loop` di theo tung round:

1. generate batch moi
2. chen mutation tu parent tot cua round truoc
3. validate local
4. chon batch da dang de gui BRAIN
5. submit/poll/import result
6. luu `submission_batches`, `submissions`, `brain_results`
7. cap nhat `alpha_history` voi `metric_source=external_brain`
8. chon parent manh dua tren BRAIN metrics that
9. tao mutation cho round tiep theo

Neu backend la `manual` va chua import result:

- round se ket thuc voi `waiting_manual_results`
- he thong khong tu dong gia lap result de chay tiep

## 4. Service pipeline 24/7

`run-service` la foreground service mode cho API backend:

```bash
python main.py run-service --config config/dev.yaml
python main.py service-status --config config/dev.yaml
```

`service-status` doc truc tiep `service_runtime`, `submission_batches`, `submissions`, va `brain_results` tu SQLite de ban xem nhanh service dang poll batch nao, con bao nhieu alpha pending, va ket qua moi nhat.

Moi tick:

1. acquire/renew single-instance DB lease
2. ensure session va refresh auth neu can
3. neu co pending jobs thi poll, normalize, persist
4. khi batch hoan tat thi update `alpha_history` va memory tu external result
5. neu khong co pending jobs thi prepare batch moi, submit, persist ngay tung job
6. ghi heartbeat vao `service_runtime`
7. sleep theo service scheduler

Resume sau restart:

- pending jobs trong `submissions` duoc poll tiep
- batch `submitting` day du metadata duoc recover thanh `submitted`
- batch `submitting` mo ho bi `paused_quarantine`
- service khong submit batch moi khi con `paused_quarantine`

Persona:

- service mode dung non-interactive auth
- neu BRAIN doi quet mat, service pause submit batch moi
- neu `service.persona_confirmation_required` bat va Telegram da cau hinh, service se doi ban xac nhan san sang truoc khi xin Persona link moi
- link Persona duoc in ra terminal va gui qua Telegram neu bot da cau hinh
- service retry auth cham, khong prompt block trong terminal

## 5. Candidate selection policy

Truoc BRAIN:

- local heuristic score
- novelty score
- prior family score
- template diversity
- field diversity
- loai near-duplicate family

Sau BRAIN:

- uu tien completed results
- uu tien `submission_eligible = true`
- rank theo `fitness`, `sharpe`, turnover chap nhan duoc
- tranh mutate qua nhieu candidate cung family

## 6. Memory update

Closed-loop update memory dua tren ket qua BRAIN that:

Positive:

- strong sharpe
- strong fitness
- acceptable turnover
- submission eligible

Negative:

- rejected
- poor fitness
- high turnover
- excessive complexity
- duplicate family no improvement

Update nay anh huong:

- template/family score
- field/operator preference
- mutation priority

## 7. Outputs

Outputs local cu:

- `outputs/generated_alphas.csv`
- `outputs/evaluated_alphas.csv`
- `outputs/selected_alphas.csv`

Outputs BRAIN manual:

- `outputs/brain_manual/brain_candidates_<batch_id>.csv`

SQLite la source traceability chinh cho:

- submission jobs
- normalized BRAIN results
- closed-loop rounds
- external memory updates
- service heartbeat / active batch / cooldown / persona wait state
