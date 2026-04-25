# 7 Turnover

## Why It Matters

Turnover khong chi la mot metric de phat hien alpha co dat submission hay khong. No cho biet alpha co dang qua nhay voi thong tin moi, co bi spike giao dich vo ich, va co con tradable khi mo rong sang universe/region khac hay khong.

## Core Takeaways

- Turnover trung binh thap khong du; spike turnover cung quan trong vi no co the giet tradability du alpha nhin dep tren trung binh.
- Giam turnover dung cach co the giu nguyen hoac con cai thien alpha neu goc van de la signal qua nhay, khong phai vi y tuong ban dau xau.
- Turnover nen duoc doc trong boi canh liquidity, universe, va chi phi giao dich, khong nen dung mot nguong co dinh cho moi profile.
- Muc tieu hop ly la toi uu ty le giua loi nhuan / thong tin va turnover, chu khong phai ep turnover xuong cang thap cang tot.

## Problem Signals

- Alpha co `sharpe` dep nhung `turnover` cao bat thuong hoac co spike rat giong nhau theo dot.
- Variant moi chi khac wrapper nhe ma turnover nhay vot len, trong khi returns khong tang tuong ung.
- Cung mot y tuong ma o universe liquid thi on, sang universe rong hon lai de tu do.
- `turnover_repair` khong bao gio kich hoat du da co nhieu parent turnover xau, hoac kich hoat nhung khong tao duoc candidate hop le.

## Apply In WQ Tool

- Map truc tiep vao `services/quality_polisher.py`, dac biet lane `turnover_repair` voi `wrap_ts_mean`, `wrap_ts_decay_linear`, `extend_existing_smoothing`.
- Khi thay bucket `low_turnover` khong ra ket qua, uu tien xem lai `recipe_guided` shapes co that su dung smoothing hay chua, thay vi voi ket luan la family do vo gia tri.
- Trong `config/brain_full.yaml`, cac knob nhu `generation.lookbacks`, `repair_policy.allow_turnover_reduction`, va objective `low_turnover` nen duoc tune cung nhau, khong tune tung muc doc lap.
- Khi doc KPI, nen tach ro `quality x turnover` thay vi chase `fitness` thuan. Alpha `fitness` kha nhung turnover tang nhanh co the la alpha kho leverage that.

## Anti-Patterns

- Giam turnover bang cach cat bo alpha rat manh roi ket luan “da on” du signal chinh da bi pha.
- Doc turnover chi qua trung binh, bo qua spike days.
- Mac dinh high turnover la loi operational. Nhieu khi do la loi design hoac wrapper choice.
- Mo rong sang less-liquid universe ma van dung cung turnover expectation cua universe liquid.

## Quick Experiments

- So sanh parent va 2-3 smoothing variants tren cung family de xem `fitness / sharpe / quality_score` giam bao nhieu khi turnover giam.
- Track rieng `turnover_repair_generated`, `turnover_repair_selected`, `turnover_repair_transform_counts` trong vai round lien tiep de xem lane nay co song hay khong.
- Kiem tra bucket `low_turnover` theo family, khong nhin tong hop chung. Neu bucket song ma selected it, van de nam o selection/crowding hon la recipe.
- Khi completed sample du, lap bang `avg_quality_score / avg_turnover` theo generation mode de tim source dang dat tradability tot hon.

## Related Repo Areas

- `services/quality_polisher.py`
- `services/kpi_report_service.py`
- `config/brain_full.yaml`
- `turnover_repair`

## Related FindingAlpha Notes

- [12_robustness.md](./12_robustness.md)
- [20_fundamental_alpha_research.md](./20_fundamental_alpha_research.md)
