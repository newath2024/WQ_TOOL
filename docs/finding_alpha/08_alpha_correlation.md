# 8 Alpha Correlation

## Why It Matters

Neu chi nhin Sharpe/Fitness ma bo qua correlation thi he thong rat de collapse vao cung mot cum y tuong. Trong repo nay, van de do xuat hien duoi dang duplicate, near-duplicate, crowding, va family concentration.

## Core Takeaways

- “Uniqueness” cua alpha la mot phan cua quality, khong phai chi la dep them cho portfolio.
- Correlation khong chi co mot kieu. Daily PnL correlation, weekly/monthly correlation, sign correlation, position correlation, va trading correlation co the ke cau chuyen khac nhau.
- Nhieu alpha nhin khac expression nhung lai dong hanh rat sat ve hanh vi. Neu khong co lop anti-collapse, batch rat de bi selection bias.
- Danh gia correlation nen gan voi use case: co luc can tranh cung PnL path, co luc can tranh cung position/trading behavior.

## Problem Signals

- `blocked_by_near_duplicate` cao lien tuc du search space nhin tren giay rat rong.
- `family_signature` hoac mot so operator paths lap lai qua nhieu trong round gan day.
- Batch co nhieu candidate selected, nhung completed results lai cung len/xuong mot kieu va quality that khong da dang.
- `family_correlation_proxy_penalty` gan nhu luon bang 0 du da thay family concentration ro rang trong report.

## Apply In WQ Tool

- Map truc tiep vao `services/selection_service.py` thong qua `family_correlation_proxy_penalty`, `family_proxy_recent_family_share`, `parent_family_overlap`, va `negative_family_surcharge`.
- `services/duplicate_service.py` va metric `blocked_by_near_duplicate` la lop chan structural truoc, con `family_correlation_proxy_penalty` la lop score sau. Doc ca hai moi biet van de nam o exact/structural hay nam o batch composition.
- `services/kpi_report_service.py` da co `by_search_bucket`, `top_search_buckets`, `negative_search_buckets`. Day la cho de doc diversity that, khong chi doc theo expression count.
- Neu mot family co `avg_quality_score` xau nhung selected van cao, uu tien xem lai pre-sim weighting thay vi tiep tuc do them budget vao family do.

## Anti-Patterns

- Dong nhat correlation voi duplicate. Near-duplicate chi bat mot phan nho cua van de.
- Thay mot alpha dep roi tiep tuc exploit quanh no den khi batch chi con mot family.
- Chi nhin “so family” ma khong nhin family nao dang chiem support va family nao dang am.
- Dung completed sample qua nho de ket luan family nay decorrelated that.

## Quick Experiments

- Lap bang `selected/generated` va `avg_quality_score` theo `family_signature` hoac `search_bucket_id`.
- So sanh rounds co `avg_family_correlation_proxy_penalty` cao hon xem selected set co da dang hon khong.
- Theo doi cung luc `blocked_by_near_duplicate` va `family_correlation_proxy_penalty` de biet nen sua duplicate guard hay selection weighting.
- Kiem tra xem winning buckets co dang keo theo mot family signature duy nhat khong. Neu co, nen giu exploration floor cho family khac.

## Related Repo Areas

- `services/selection_service.py`
- `services/duplicate_service.py`
- `services/kpi_report_service.py`
- `family_correlation_proxy_penalty`
- `blocked_by_near_duplicate`

## Related FindingAlpha Notes

- [11_triple_axis_plan.md](./11_triple_axis_plan.md)
- [15_automated_search.md](./15_automated_search.md)
