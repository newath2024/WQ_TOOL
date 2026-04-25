# 11 The Triple-Axis Plan

## Why It Matters

TAP la cach don gian de tranh search bi lech vao mot goc nho cua alpha space. No rat hop voi repo nay vi `recipe_guided` da co `search_bucket`, `recipe_family`, va `objective_profile` - gan nhu chinh la mot phien ban operational cua TAP.

## Core Takeaways

- Alpha space nen duoc nhin qua 3 truc: `ideas & datasets`, `regions & universes`, `performance parameters`.
- Diversity that khong phai la random thuần; no la coverage co to chuc tren cac truc quan trong.
- TAP giup phat hien “cho trong” trong portfolio/search space, thay vi cu mo rong mot huong da quen.
- Chon focus theo trục la cach tot de mo rong search space ma van giu duoc discipline.

## Problem Signals

- Search space co ve rong nhung ket qua that tap trung vao 1-2 motif/family.
- Moi patch moi deu la them transform, nhung khong ai tra loi duoc ta dang thieu dataset/family/objective nao.
- `recipe_guided` co nhieu bucket, nhung active buckets khong duoc xem nhu mot portfolio coverage problem.
- Nhom alpha dep chi den tu `balanced`, con `quality` va `low_turnover` gan nhu khong co vai tro.

## Apply In WQ Tool

- `search_bucket_id = recipe_family|fundamental|objective_profile` la implementation hook ro nhat cua TAP trong repo hien tai.
- `recipe_family` la đại dien cho truc `idea/dataset`, `objective_profile` la truc performance, con region/universe co the mo rong sau qua profile/config thay vi nhoi het vao V1.
- Khi them family moi, nen hoi: no them truc nao vao TAP? Neu cau tra loi la “chi them mot wrapper nua” thi kha nang cao la patch do khong mo rong alpha space that.
- `docs/next_round_budget_recommendation_2026-04-24.md` nen duoc doc nhu mot TAP dashboard thu cong: bucket nao dang duoc khai thac qua nhieu, bucket nao dang bi bo trong.

## Anti-Patterns

- The same idea everywhere: cung mot family nhung doi wrapper/params va goi do la diversity.
- Tang budget cho bucket thang ma khong giu exploration floor cho bucket khac.
- Bo qua objective-specific variants roi ket luan family nao cung phai optimize theo mot kieu.
- Dua qua nhieu dataset/family vao mot alpha, lam signal mat focus.

## Quick Experiments

- Lap bang active bucket coverage theo 3-5 rounds gan nhat de xem truoc sau patch TAP co rong hon that khong.
- Kiem tra `by_search_bucket` trong KPI report de tim bucket co `selected/generated` cao nhung support completed con thap.
- Khi tao family moi, viet truoc 1 dong “truc nao duoc mo rong” trong docs/PR note.
- So sanh `balanced` vs `quality` vs `low_turnover` trong cung family de xem objective profile nao that su co y nghia.

## Related Repo Areas

- `services/recipe_guided_generator.py`
- `services/brain_batch_service.py`
- `search_bucket`
- `recipe_family`
- `objective_profile`

## Related FindingAlpha Notes

- [15_automated_search.md](./15_automated_search.md)
- [20_fundamental_alpha_research.md](./20_fundamental_alpha_research.md)
