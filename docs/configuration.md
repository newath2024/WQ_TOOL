# Configuration

## Profiles

- `dev`: iterate nhanh trên sample dataset
- `research`: profile mặc định, thresholds hợp lý hơn cho local screening
- `strict`: shortlist chặt hơn trước khi review thủ công

## Schema mới

```yaml
evaluation:
  hard_filters:
    min_validation_sharpe: 0.25
    max_validation_turnover: 3.0
    max_validation_drawdown: 0.50
  data_requirements:
    min_validation_observations: 15
    min_stability: 0.20
  diversity:
    signal_correlation_threshold: 0.90
    returns_correlation_threshold: 0.90
  ranking:
    top_k: 20
    use_behavioral_novelty_tiebreak: true
  robustness:
    enable_subuniverse_test: true
    enable_ladder_test: true
    enable_robustness_test: true
    ladder_buckets: 3
    ladder_min_sharpe: 0.0
    ladder_min_passes: 2
    subuniverse_min_sharpe: -0.10
    subuniverse_min_pass_fraction: 0.60
    robustness_min_fitness_ratio: 0.10
```

## Backward compatibility

Schema cũ vẫn hợp lệ:

- `evaluation.min_sharpe`
- `evaluation.max_turnover`
- `evaluation.min_observations`
- `evaluation.max_drawdown`
- `evaluation.min_stability`
- `evaluation.signal_correlation_threshold`
- `evaluation.returns_correlation_threshold`
- `evaluation.top_k`
- `submission_tests.*`

Loader sẽ normalize nội bộ sang grouped config.

## Nguyên tắc chọn profile

- dùng `dev` khi đang sửa logic hoặc debug
- dùng `research` khi muốn shortlist có ý nghĩa hơn
- dùng `strict` khi muốn giảm false positives trước khi xem thủ công

## Lưu ý về sample data

Bundled sample data chỉ giúp:

- verify command wiring
- chạy smoke tests
- kiểm tra persistence, memory, lineage

Không nên suy luận chất lượng alpha thật từ sample dataset này.
