# 31 Introduction to WebSim

## Why It Matters

Chapter nay quan trong khong vi UI lich su cua WebSim, ma vi no dat lai mindset dung: co mot simulation platform, co user input, co settings, co feedback, va co knowledge loop. `WQ Tool` dang lam mot phien ban workflow-oriented cua tu duy do quanh BRAIN service va closed-loop pipeline.

## Core Takeaways

- Market simulation platform co gia tri lon khi no khong chi cho ket qua, ma con day nguoi dung cach nghi ve alpha ideas va cach doc feedback.
- Idea sourcing la mot phan cua workflow: papers, journals, blogs, technical indicators, va data themes deu la nguon input hop le.
- Managing simulation settings va analyzing results la ky nang cot loi, khong phai viec phu sau khi da co expression.
- Knowledge base + educational layer giup bien viec test alpha thanh vong lap hoc nhanh hon.

## Problem Signals

- Moi lan gap issue, team phai “nho lai” ly thuyet tu dau thay vi co playbook de mo ngay.
- Chay service duoc nhung khong co mot cho tong hop de hoi “van de nay nen nghi theo huong nao?”.
- Nguon y tuong mo rong alpha space phu thuoc qua nhieu vao tri nho hoac mot vai prompt hoc thuoc.
- Nguoi moi vao repo doc code xong van khong biet bat dau tu problem framing nao.

## Apply In WQ Tool

- Folder `docs/finding_alpha/` chinh la lop knowledge/education nhe de bo tro cho closed-loop research workflow.
- `README.md` cua folder nay nen duoc coi la diem vao truoc khi design patch moi cho quality, turnover, correlation, hay recipe generation.
- Khi gap van de moi, nen map no sang mot “problem type” truoc, roi moi map sang file code. Day la tinh than WebSim-style learning loop.
- `run-service`, KPI report, budget recommendation, va knowledge notes hop lai thanh feedback system, thay vi de tung artifact song rieng le.

## Anti-Patterns

- Xem simulation platform chi la may cham diem expression.
- Tiep can alpha research chi bang trial-and-error code ma khong co knowledge loop.
- Lap lai cung mot sai lam vi khong co mot cho tra cuu mental model nhanh.
- Doc WebSim theo huong san UI features ma bo qua workflow learning duoi nen.

## Quick Experiments

- Moi khi de xuat patch, ghi ro note do dang phan loai van de theo file nao trong `docs/finding_alpha/`.
- Sau moi vong test, cap nhat budget recommendation hoac note van de bang language cua playbooks, khong chi bang metric raw.
- Kiem tra xem issue moi co that su moi khong, hay da thuoc mot nhom cu nhu `turnover`, `correlation`, `overfitting`, `search-space organization`.
- Dung folder nay nhu onboard pack nhe cho future agents truoc khi cho phep sua quality pipeline.

## Related Repo Areas

- `docs/finding_alpha/README.md`
- `progress_logs/`
- `services/kpi_report_service.py`
- `services/brain_batch_service.py`

## Related FindingAlpha Notes

- [09_backtest_signal_or_overfitting.md](./09_backtest_signal_or_overfitting.md)
- [15_automated_search.md](./15_automated_search.md)
