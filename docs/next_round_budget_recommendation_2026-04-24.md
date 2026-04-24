# Next-Round Budget Recommendation

Run: `8c1f78cc6618`  
Date: `2026-04-24`  
Data scope:
- Baseline: rounds `12545-12555`
- Recent V2 window: rounds `12556-12566`
- Round `12567` excluded from quality judgment because service stopped in `waiting_persona_confirmation` with `7` pending jobs

## Executive Summary

Recent V2 rounds are better than baseline on the main quality metrics:

| KPI | Baseline | Recent |
|---|---:|---:|
| Completed rate | 24.3% | 36.0% |
| Timeout rate | 75.7% | 64.0% |
| Avg fitness | 0.0544 | 0.1070 |
| Avg sharpe | 0.1014 | 0.1387 |
| Avg quality score | 0.0222 | 0.0352 |

The main new winner is `recipe_guided`, not `quality_polish`:

| Source | Raw | Completed | Completed rate | Avg fitness | Avg sharpe | Avg quality |
|---|---:|---:|---:|---:|---:|---:|
| `fresh` | 84 | 29 | 34.5% | 0.0797 | 0.1097 | 0.0203 |
| `quality_polish` | 4 | 1 | 25.0% | 0.4200 | 0.4200 | 0.1974 |
| `recipe_guided` | 17 | 7 | 41.2% | 0.1843 | 0.2529 | 0.1062 |

Important caveats:
- `quality_polish` quality is promising but sample is still tiny.
- `turnover_repair` has not triggered yet: `turnover_repair_generated = 0` across rounds `12556-12566`.
- `dynamic budget` is alive, but it is still mostly neutral because support is thin. Source and bucket allocations are not yet adapting aggressively.

## What To Increase / Reduce Now

### Source-Level Recommendation

Global recommendation for the next few rounds:

| Source | Recommendation | Reason |
|---|---|---|
| `recipe_guided` | Increase when strong buckets are active | Best realized quality so far |
| `fresh` | Trim slightly, but keep as main volume source | Still carries most throughput and acceptable quality |
| `quality_polish` | Keep flat for now | Good quality, but too little output to justify a big raise |

Suggested post-mutation target mix:

| Situation | `fresh` | `quality_polish` | `recipe_guided` |
|---|---:|---:|---:|
| Weak recipe bucket cycle | 40-42 | 22-24 | 12-16 |
| Neutral cycle | 36-38 | 22-24 | 18-22 |
| Strong recipe bucket cycle | 30-34 | 20-24 | 24-28 |

### Bucket-Level Recommendation

#### Increase

| Bucket | Evidence |
|---|---|
| `revision_surprise|fundamental|balanced` | `avg_fitness 0.33`, `avg_sharpe 0.49`, `2/3` completed |
| `revision_surprise|fundamental|quality` | `avg_fitness 0.32`, `avg_sharpe 0.48`, strong selection hit rate |
| `fundamental_quality|fundamental|balanced` | `avg_fitness 0.1567`, `avg_sharpe 0.21`, `3/4` completed |

#### Keep Small Exploration

| Bucket | Evidence |
|---|---|
| `value_vs_growth|fundamental|balanced` | `2/3` selected into sim, but still no completed results |

#### Reduce To Floor / Watch

| Bucket | Evidence |
|---|---|
| `fundamental_quality|fundamental|quality` | negative realized quality: `avg_fitness -0.16`, `avg_sharpe -0.32` |
| `accrual_vs_cashflow|fundamental|balanced` | no selected, no raw results yet |
| `accrual_vs_cashflow|fundamental|quality` | no selected, no raw results yet |
| `value_vs_growth|fundamental|quality` | no selected, no raw results yet |
| `*_low_turnover` buckets | no proof yet; keep exploration only |

## Active Buckets For Upcoming Rounds

The bucket scheduler rotates deterministically.

### Round `12568`

Active buckets:
- `accrual_vs_cashflow|fundamental|quality`
- `accrual_vs_cashflow|fundamental|low_turnover`
- `value_vs_growth|fundamental|balanced`
- `value_vs_growth|fundamental|quality`

Recommendation:
- This is a weak cycle.
- Do **not** blindly raise `recipe_guided` here.

Suggested split if you want to guide it manually:

| Bucket | Suggested candidates |
|---|---:|
| `value_vs_growth|fundamental|balanced` | 6 |
| `accrual_vs_cashflow|fundamental|quality` | 2 |
| `accrual_vs_cashflow|fundamental|low_turnover` | 2 |
| `value_vs_growth|fundamental|quality` | 2 |

Source mix suggestion for this round:
- `fresh 40-42`
- `quality_polish 22-24`
- `recipe_guided 12-16`

### Round `12569`

Active buckets:
- `value_vs_growth|fundamental|low_turnover`
- `revision_surprise|fundamental|balanced`
- `revision_surprise|fundamental|quality`
- `revision_surprise|fundamental|low_turnover`

Recommendation:
- This is the strongest upcoming cycle.
- This is the best round to push `recipe_guided`.

Suggested split:

| Bucket | Suggested candidates |
|---|---:|
| `revision_surprise|fundamental|balanced` | 9-10 |
| `revision_surprise|fundamental|quality` | 8-9 |
| `revision_surprise|fundamental|low_turnover` | 3 |
| `value_vs_growth|fundamental|low_turnover` | 2 |

Source mix suggestion for this round:
- `fresh 30-34`
- `quality_polish 20-24`
- `recipe_guided 24-28`

### Round `12570`

Active buckets:
- `fundamental_quality|fundamental|balanced`
- `fundamental_quality|fundamental|quality`
- `fundamental_quality|fundamental|low_turnover`
- `accrual_vs_cashflow|fundamental|balanced`

Recommendation:
- Mixed cycle.
- Favor `fundamental_quality|balanced`, keep the rest lean.

Suggested split:

| Bucket | Suggested candidates |
|---|---:|
| `fundamental_quality|fundamental|balanced` | 8 |
| `fundamental_quality|fundamental|quality` | 4 |
| `fundamental_quality|fundamental|low_turnover` | 2 |
| `accrual_vs_cashflow|fundamental|balanced` | 2 |

Source mix suggestion for this round:
- `fresh 36-40`
- `quality_polish 20-24`
- `recipe_guided 16-20`

## What The Current Code Will Actually Do

Important: the current V2 code does **not** yet enforce the manual bucket splits above.

What it will do now:
- keep `source_budget_allocations` alive
- keep `recipe_bucket_budget_allocations` alive
- apply `family_correlation_proxy_penalty`
- but still stay close to neutral allocations while support is thin

Observed recent behavior:
- `source_budget_allocations` stayed near `fresh 38`, `quality_polish 24-25`, `recipe_guided 20`
- `recipe_bucket_budget_allocations` stayed near `5/5/5/5`

So this report is a **decision report**, not a claim that the tool will already follow these bucket weights automatically.

## Immediate Recommendation

If the goal is to run the next test round without another code patch:
- do not expect `12568` to be the best evidence round for recipe buckets
- treat `12569` as the real proving round for `recipe_guided`
- keep watching:
  - `recipe_guided_selected/generated`
  - realized results from `revision_surprise` buckets
  - whether `quality_polish` starts producing more than `0-2` candidates per round
  - whether `turnover_repair` ever activates

If the goal is to make the tool follow this budget advice more closely, the next patch should add:
- manual per-bucket budget bias or override
- stronger promotion for winning buckets
- stronger demotion for losing buckets even before support reaches the current neutral threshold
