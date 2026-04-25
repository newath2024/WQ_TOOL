# 12 Techniques for Improving the Robustness of Alphas

## Why It Matters

Nhieu patch quality that ra khong vi tim duoc y tuong moi, ma vi biet bien signal thanh robust hon. Chapter nay rat khop voi repo vi wrappers nhu `rank`, `zscore`, `ts_mean`, va tu duy trim/winsorize/outlier control da xuat hien lien tuc trong quality tuning.

## Core Takeaways

- Robust methods giam do nhay cam voi outlier va extreme values, nhat la khi signal goc khong on dinh.
- `rank`, `zscore`, trimming, winsorizing, va smoothing la cac cach chuan hoa/lam min signal giup signal stable hon.
- Robustness khong chi de “lam dep” metric; no giup alpha song sot tot hon khi market regime thay doi.
- Wrapper dung co the tang kha nang transfer cua y tuong sang universe/region/profile khac.

## Problem Signals

- Alpha chi dep trong mot window nho, ra khoi window do la vo.
- `validation_invalid_nesting` giam nhung result that van loạn, cho thay wrapper stack co the van chua dung loai.
- Cung mot y tuong, raw variant xau nhung ban `rank(ts_mean(...))` on dinh hon ro ret.
- Metrics dao dong manh theo outlier rounds hoac chi do mot vai points extreme.

## Apply In WQ Tool

- Trong `quality_polish`, uu tien wrappers co tinh robust nhu `wrap_rank`, `wrap_zscore`, va `window_perturb` truoc khi them transforms manh tay hon.
- `recipe_guided_generator` da dung `rank`, `ts_mean`, `ts_delta` cho objective-specific drafts; day la noi nen doc chapter nay truoc khi them recipe shape moi.
- Khi local validation fail tang cao, dung ep them operator la. Thuong huong dung la chon shape robust va don gian hon truoc.
- Khi completed sample cho thay signal raw on dinh kem, uu tien them robust layer vao winning family thay vi tim parent moi ngau nhien.

## Anti-Patterns

- Them wrapper chong layer ma khong ro no giai quyet outlier hay chi lam expression phuc tap hon.
- Dung `zscore` hay `rank` nhu mot nghi le mac dinh cho moi alpha ma khong xem no co phu hop voi objective khong.
- Nham robustification voi lam muon qua muc, den muc alpha mat het signal.
- Bo qua cac ky thuat limit outlier chi vi local runtime khong mo phong du BRAIN.

## Quick Experiments

- So sanh theo family: `raw`, `rank(raw)`, `rank(ts_mean(raw,d))`, `zscore(ts_mean(raw,d))`.
- Theo doi xem wrapper nao lam `selected/generated` tang ma khong day `blocked_by_near_duplicate` tang qua nhanh.
- Kiem tra cac winner gan day xem co mau so chung nao ve wrapper/normalization khong.
- Khi mot family bi noisy, test tang lookback/smoothing truoc khi loai bo ca family.

## Related Repo Areas

- `services/quality_polisher.py`
- `services/recipe_guided_generator.py`
- `generation.lookbacks`
- `wrap_rank`
- `ts_mean`

## Related FindingAlpha Notes

- [07_turnover.md](./07_turnover.md)
- [15_automated_search.md](./15_automated_search.md)
