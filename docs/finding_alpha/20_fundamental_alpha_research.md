# 20 Fundamental Analysis and Alpha Research

## Why It Matters

Chapter nay quan trong vi repo dang di theo huong `fundamental-heavy recipe generation`. Sach nhac rang financial statements, accruals, cash flow, profitability, debt, va analyst-driven expectation changes la nguon goc rat hop ly cho alpha - neu biet to chuc thanh family ro rang.

## Core Takeaways

- Financial statements cho phep tao factor/alpha quanh profitability, debt burden, liquidity, cash generation, va red flags trong working capital.
- Cash flow va quality of earnings thuong giai thich duoc nhieu hon la chi nhin level accounting thuan.
- Inventory, receivables, reserves, debt-equity, va cac dau hieu suy yeu tren statements la nguon y tuong tot cho red-flag style alphas.
- Fundamental ideas manh hon khi duoc viet thanh so sanh co cau truc nhu `cashflow vs accrual`, `value vs growth`, `surprise vs expectation`.

## Problem Signals

- Bucket fundamental co selected rate kha nhung completed quality that khong on, cho thay field pairing hoac recipe shape chua dung.
- `fundamental_quality|quality` cho ket qua am, trong khi `fundamental_quality|balanced` lai on, cho thay objective profile co the dang over-constrain.
- Family fundamental dang dung field level don le qua nhieu, thieu cap so sanh co y nghia giua hai nhom field.
- Alpha fundamental dep o local logic nhung khong lap lai tren BRAIN, cho thay can chuan hoa/robustify hon la them fields ngau nhien.

## Apply In WQ Tool

- `services/recipe_guided_generator.py` da co 4 family hop ly voi chapter nay: `fundamental_quality`, `accrual_vs_cashflow`, `value_vs_growth`, `revision_surprise`.
- Khi them recipe moi, nen uu tien pair co y nghia kinh te ro: profitability vs weakness, cash generation vs accrual, value vs expectation, revision vs stale consensus.
- `revision_surprise` dang co tin hieu tot trong data gan day; chapter nay la ly do de tiep tuc khai thac expectation-change themes, khong chi mo rong profitability.
- Field picking nen dung keyword va category de giu family focus, tranh tron qua nhieu field khong cung “cau chuyen”.

## Anti-Patterns

- Nem tat ca field fundamental vao mot alpha va hy vong search tu biet chon.
- Chi nhin level cua mot ratio ma khong so sanh voi mot nhom field doi lap.
- Overfit fundamental quality bang cach them wrapper lien tuc quanh mot field dep duy nhat.
- Dong nhat “fundamental” voi “low turnover”; hai cai co lien quan nhung khong phai luc nao cung trung.

## Quick Experiments

- Xep hang ket qua theo family fundamental va objective profile, khong gom chung thanh “recipe_guided”.
- Tao bang `selected/generated` cho `fundamental_quality`, `accrual_vs_cashflow`, `value_vs_growth`, `revision_surprise`.
- So sanh recipe don field voi recipe doi lap hai truong de xem loai nao song tot hon tren BRAIN.
- Khi bucket quality xau nhung balanced on, thu giam do “quality bias” truoc khi bo ca family.

## Related Repo Areas

- `services/recipe_guided_generator.py`
- `search_bucket`
- `recipe_family`
- `objective_profile`

## Related FindingAlpha Notes

- [11_triple_axis_plan.md](./11_triple_axis_plan.md)
- [15_automated_search.md](./15_automated_search.md)
