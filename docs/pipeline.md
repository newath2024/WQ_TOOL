# Pipeline

## 1. Data load

- load OHLCV và metadata phụ trợ
- normalize schema
- validate keys, timestamp order, duplicate rows, price sanity
- lưu dataset summary và dataset fingerprint vào `runs`

## 2. Candidate generation

- template / grammar / guided / mutation branches
- guided branches dùng `PatternMemorySnapshot` của đúng regime
- generation metadata lưu mode, source patterns, source genes, parent refs

## 3. Expression evaluation

- parse AST
- validate field/operator/depth/group usage
- evaluate ra wide signal matrix

## 4. Simulation

- signal controls
- neutralization
- optional volatility scaling
- portfolio construction
- transaction costs
- net return, turnover, cumulative return

## 5. Split metrics

- compute train / validation / test metrics riêng
- validation là driver chính cho filtering, ranking, adaptive memory
- test chỉ để audit

## 6. Filtering và ranking

- hard filters
- data sufficiency
- submission-style robustness checks
- dedup theo signal/returns correlation
- ranking theo fitness -> submission -> sharpe -> novelty -> complexity

## 7. Persistence

- metrics, selections, submission tests
- adaptive history, diagnoses, pattern memberships
- simulation cache
- run metadata và config snapshot

## 8. Iteration

- `report` xem chất lượng run
- `memory-*` xem structural signals
- `lineage` xem ancestry
- `mutate` tạo vòng tiếp theo
