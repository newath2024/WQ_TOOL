# 15 Alphas from Automated Search

## Why It Matters

Day la chapter sat nhat voi `WQ Tool` hien tai. No nhac rang chat luong cua automated search khong nen duoc doc qua tung alpha le, ma qua `batch performance`, search space design, va discipline cua vong lap generate -> test -> optimize.

## Core Takeaways

- Trong automated search, confidence thuong nam o batch va search process, khong nam o mot alpha le khong co explanation ro.
- Search nen di theo coarse-to-fine, nhe o vong dau va giai quyet tinh tin cay / robustness o vong sau.
- Search space chat luong quan trong ngang bang scoring algorithm; search space xau se cho batch xau du scoring co ve thong minh.
- Selection bias rat de xay ra neu sau khi search xong ta chi nhin cac winner dep nhat.

## Problem Signals

- `attempt_success_count` on nhung completed quality khong tang, cho thay search space co the dang phung to vo ich.
- Co vai alpha dep, nhung `selected/generated` va `completed/raw` cua ca source hoac bucket khong cai thien.
- Dynamic budget song nhung van gan neutral qua lau, cho thay support that chua du hoac source chua co yield du ro.
- Batch moi chi tao them “ban sao bien the” quanh cung mot parent, thay vi mo rong co to chuc.

## Apply In WQ Tool

- `services/brain_batch_service.py` la noi tap hop toan bo automated search pipeline: `mutation -> quality_polish -> recipe_guided -> fresh`.
- `quality_polish` nen duoc doc nhu exploit lane co kiem soat; `recipe_guided` la organized search lane; `fresh` la exploration floor.
- `source_budget_allocations`, `source_yield_scores`, `recipe_bucket_budget_allocations`, va `recipe_bucket_yield_scores` la batch-performance telemetry. Dung nhung cai nay de phan xet search, khong chi nhin top alpha.
- Khi mot source/bucket thang, uu tien hoi “search space nao dang dung?” truoc khi hoi “alpha nao dang dung?”.

## Anti-Patterns

- Sau moi round, chi pick mot winner dep nhat roi toan bo pipeline quay quanh no.
- Tang budget cho source moi chi vi 1 completed result dep.
- Dong nhat validation success voi quality success.
- De support qua thap nhung van ep dynamic budget reallocate manh.

## Quick Experiments

- Bao cao 5-10 rounds theo `source`: `generated`, `selected`, `completed`, `avg_quality_score`, `positive_fitness_rate`.
- Bao cao 5-10 rounds theo `bucket` de xac dinh search space nao dang co batch yield that.
- Khi patch source moi, danh gia no bang `selected/generated` va `completed_rate` truoc, roi moi danh gia top winner.
- Neu dynamic budget neutral qua lau, giam support threshold hoac them manual bias tam thoi cho winning buckets.

## Related Repo Areas

- `services/brain_batch_service.py`
- `services/quality_polisher.py`
- `services/recipe_guided_generator.py`
- `source_budget_allocations`
- `source_yield_scores`

## Related FindingAlpha Notes

- [09_backtest_signal_or_overfitting.md](./09_backtest_signal_or_overfitting.md)
- [11_triple_axis_plan.md](./11_triple_axis_plan.md)
