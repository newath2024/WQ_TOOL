# 9 Backtest - Signal or Overfitting?

## Why It Matters

Khi BRAIN chay cham hoac completed sample rat nho, repo rat de roi vao bay “survivor reading”: thay vai alpha song sot dep roi tuong code vua cai tien that. Chapter nay nhac rang phai tach quality signal that khoi noise, selection bias, va operational distortion.

## Core Takeaways

- Ket qua dep co the den tu mo phong phi thuc te, bias, hoac random chance; ban than so dep khong tu dong xac nhan co predictive power.
- Overfitting thuong xuat hien khi ta gap qua nhieu thuoc do va qua it ky luat giai thich underlying driver.
- Selection bias dac biet nguy hiem trong automated search vi con nguoi rat de nho alpha dep va quen alpha chet.
- Khi sample nho, dieu quan trong hon la kha nang lap lai va consistency theo batch/window, khong phai 1-2 winner.

## Problem Signals

- `completed` rat it nhung `avg_fitness` hoac `avg_sharpe` nhin rat dep, de lam minh tuong patch da thanh cong.
- Timeout/ops issue lam raw results rat cao nhung completed rat lech, trong khi report chi nhin surviving winners.
- Sau moi lan doi search space, mot vai alpha dep xuat hien nhung batch yield khong tang hoac khong lap lai.
- Metrics tong dep hon baseline chi nhờ mot alpha outlier.

## Apply In WQ Tool

- Khi danh gia patch, uu tien so `recent window` voi `baseline` theo completed rate, avg quality, positive rate, va batch-level yield; dung chi nhin top alpha.
- `poll_timeout_after_downtime` can duoc xem la operational issue, khong phai negative alpha signal. Doc report theo 2 tang: ops va quality.
- `services/kpi_report_service.py` va `docs/next_round_budget_recommendation_2026-04-24.md` nen duoc doc cung nhau de tranh ket luan sai tu sample nho.
- Neu `recipe_guided` co 1-2 winner dep nhung support qua mong, nen tang tu tu va doi them rounds, khong nhay thang vao exploit mạnh.

## Anti-Patterns

- Thay 1 winner dep roi ket luan search space nay chac chan dung.
- Dung timeout cao de ket luan alpha xau, trong khi backend dang co van de.
- Chon chi nhung alpha out-of-sample dep nhat sau automated search roi goi do la batch quality.
- Quen mat cost / tradability / turnover khi chi nhin Sharpe.

## Quick Experiments

- Bao cao moi patch theo hai cua so: `recent completed` va `recent raw`, de thay ro selection distortion.
- Doc `median` cung voi `avg` cho fitness, sharpe, quality_score de phat hien outlier-driven improvement.
- So `selected/generated` va `completed/raw` theo source (`fresh`, `quality_polish`, `recipe_guided`) thay vi chi doc tong.
- Khi completed sample < 10 cho mot source/bucket, danh dau ket luan la “promising” thay vi “confirmed”.

## Related Repo Areas

- `services/kpi_report_service.py`
- `services/brain_batch_service.py`
- `brain_results`
- `poll_timeout_after_downtime`

## Related FindingAlpha Notes

- [15_automated_search.md](./15_automated_search.md)
- [31_websim.md](./31_websim.md)
