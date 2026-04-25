# Alpha Performance and Quality Analysis

Run: `8c1f78cc6618`  
Database: `dev_wq_tool.sqlite3`  
Log: `progress_logs/8c1f78cc6618.jsonl`  
Analysis date: `2026-04-25`  

Notes:
- DB/log timestamps are stored in UTC.
- Latest observed service log: `2026-04-25T01:00:42+00:00`.
- Latest round: `12605`.
- Round `12605` is still not terminal: `1` BRAIN job remains `running`.
- BRAIN check interpretation is conservative: many checks remain `PENDING`; counts below for "no FAIL checks" do not imply accepted submission.

## Executive Summary

The system has improved materially since the previous decision window, but the main blocker is still BRAIN-side throughput and submission-test quality.

| Window | Terminal BRAIN jobs | Completed | Completed rate | Avg Sharpe | Avg Fitness | Avg Quality | Positive quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline `12545-12555` | 115 | 28 | 24.3% | 0.1014 | 0.0544 | 0.0222 | 64.3% |
| Recent V2 `12556-12566` | 111 | 40 | 36.0% | 0.1387 | 0.1070 | 0.0352 | 62.5% |
| Post-report `12567-12605` | 417 | 187 | 44.8% | 0.1767 | 0.0891 | 0.0449 | 65.2% |
| Latest 25 `12581-12605` | 264 | 118 | 44.7% | 0.1917 | 0.0991 | 0.0585 | 69.5% |
| Latest 10 `12596-12605` | 108 | 50 | 46.3% | 0.2318 | 0.1224 | 0.0768 | 72.0% |

Main takeaways:
- Completion rate has improved from `24.3%` baseline to `44-46%` recently.
- Quality trend is up: latest 10 rounds have the best average Sharpe, fitness, quality, and positive-quality rate.
- `recipe_guided` is now consistently better than fresh generation on realized BRAIN quality.
- `quality_polish` has the best realized quality, but the sample is tiny and output has dried up in the latest 10 rounds.
- No recent alpha is clean enough for confidence: `LOW_SHARPE` and `LOW_FITNESS` dominate BRAIN check failures.

## Current State

Latest service runtime:

| Field | Value |
|---|---|
| Service status | `service_stopped_pending` |
| Active batch | `brain-8c1f78cc-r12605-13d934c3` |
| Pending jobs | `1` |
| Last heartbeat | `2026-04-25T01:00:42+00:00` |
| Last success | `2026-04-24T22:53:17+00:00` |
| Last error | `Waiting for Telegram confirmation before requesting a new Persona link.` |

Open running submission:

| Round | Candidate | Submitted | Last polled | Timeout deadline | Expression |
|---:|---|---|---|---|---|
| `12605` | `61b296f8de200389` | `2026-04-24T22:44:05+00:00` | `2026-04-24T22:52:53+00:00` | `2026-04-25T00:55:40+00:00` | `rank(((ts_mean(close,100)*ts_decay_linear(anl69_roa_best_eeps_nxt_yr,100))*ts_mean(anl69_roa_best_eeps_nxt_yr,100)))` |

The timeout deadline is already before the final heartbeat, but the submission remains marked `running`; this is a stale job/state cleanup issue to watch.

## Pipeline Throughput

| Window | Rounds | Generated | Validated | Submitted | Completed in round aggregate | Validation rate | Submit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| All closed-loop rows | 1,178 | 68,535 | 48,835 | 4,933 | 4,859 | 71.3% | 7.2% |
| Baseline `12545-12555` | 11 | 1,100 | 645 | 115 | 115 | 58.6% | 10.5% |
| Recent V2 `12556-12566` | 11 | 1,100 | 660 | 111 | 111 | 60.0% | 10.1% |
| Post-report `12567-12605` | 39 | 3,900 | 2,350 | 418 | 417 | 60.3% | 10.7% |
| Latest 25 `12581-12605` | 25 | 2,500 | 1,524 | 265 | 264 | 61.0% | 10.6% |
| Latest 10 `12596-12605` | 10 | 1,000 | 635 | 109 | 108 | 63.5% | 10.9% |

Interpretation:
- Local validation is stable around `60-64%` recently.
- Submit rate is stable around `10-11%`.
- The actual bottleneck is after submission: BRAIN completion rate is only `44-46%` recently because timeout and auth/persona interruptions remain frequent.

Post-report generation stage performance:

| Metric | Value |
|---|---:|
| Generation rounds | 39 |
| Avg generation time | 29.4s |
| Avg attempts per round | 141.6 |
| Avg generated successes per round | 100.0 |
| Avg validation failures per round | 2.9 |
| Avg duplicate failures per round | 27.7 |

## Source-Level Quality

Post-report realized BRAIN quality:

| Source | Terminal | Completed | Completion rate | Avg Sharpe | Avg Fitness | Avg Quality | Positive quality | Avg Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `quality_polish` | 2 | 2 | 100.0% | 0.8550 | 0.6450 | 0.3623 | 100.0% | 0.0162 |
| `recipe_guided` | 32 | 31 | 96.9% | 0.3213 | 0.1584 | 0.1496 | 90.3% | 0.0418 |
| `fresh` | 383 | 154 | 40.2% | 0.1388 | 0.0678 | 0.0197 | 59.7% | 0.0724 |

All-time realized BRAIN quality:

| Source | Terminal | Completed | Completion rate | Avg Sharpe | Avg Fitness | Avg Quality | Positive quality | Avg Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `quality_polish` | 5 | 5 | 100.0% | 0.8580 | 0.5920 | 0.1844 | 60.0% | 0.0468 |
| `recipe_guided` | 41 | 38 | 92.7% | 0.3087 | 0.1632 | 0.1416 | 86.8% | 0.0366 |
| `fresh` | 5,838 | 2,148 | 36.8% | 0.0874 | 0.0346 | 0.0019 | 6.0% | 0.1214 |

Interpretation:
- `recipe_guided` is no longer just promising; it is clearly better than `fresh` on completion rate and quality.
- `quality_polish` is the highest-quality source, but only `2` post-report completions and `5` all-time completions are too small for aggressive budget expansion.
- `fresh` still supplies most candidates and occasional top outliers, but its average quality remains weak.

## Source Budget and Selection

Post-report generation source metrics:

| Source | Generated | Selected |
|---|---:|---:|
| `fresh` | 3,047 | 336 |
| `recipe_guided` | 260 | 57 |
| `quality_polish` | 8 | 3 |
| `mutation` | n/a | 22 |

Latest 10 rounds:

| Source | Generated | Selected |
|---|---:|---:|
| `fresh` | 650 | 60 |
| `recipe_guided` | 200 | 44 |
| `quality_polish` | 0 | 0 |
| `mutation` | n/a | 5 |

Interpretation:
- The scheduler has effectively ramped `recipe_guided` up in the latest 10 rounds.
- `recipe_guided` selected rate is strong: `44/200 = 22.0%` latest 10, versus `60/650 = 9.2%` for fresh.
- `quality_polish` is not producing in the latest 10 rounds despite having high realized quality. This is likely parent/transform exhaustion or cooldown, not evidence that the idea is bad.
- `turnover_repair` still has not activated: generated `0`, selected `0`.

## Recipe Bucket Quality

Post-report recipe bucket results:

| Bucket | Terminal | Completed | Avg Quality | Avg Sharpe | Avg Fitness | Avg Turnover |
|---|---:|---:|---:|---:|---:|---:|
| `fundamental_quality|fundamental|quality` | 3 | 3 | 0.2634 | 0.6900 | 0.2967 | 0.0355 |
| `fundamental_quality|fundamental|low_turnover` | 4 | 4 | 0.2451 | 0.5425 | 0.3175 | 0.0195 |
| `revision_surprise|fundamental|quality` | 1 | 1 | 0.2109 | 0.4800 | 0.2300 | 0.0721 |
| `value_vs_growth|fundamental|balanced` | 4 | 4 | 0.1999 | 0.4725 | 0.2150 | 0.1061 |
| `revision_surprise|fundamental|balanced` | 6 | 6 | 0.1787 | 0.3683 | 0.1733 | 0.0413 |
| `revision_surprise|fundamental|low_turnover` | 6 | 6 | 0.1165 | 0.2450 | 0.1300 | 0.0162 |
| `value_vs_growth|fundamental|quality` | 2 | 2 | 0.1023 | 0.1000 | 0.0300 | 0.0374 |
| `accrual_vs_cashflow|fundamental|balanced` | 3 | 3 | 0.0895 | 0.0467 | 0.0100 | 0.0473 |
| `accrual_vs_cashflow|fundamental|quality` | 3 | 2 | -0.1933 | -0.3350 | -0.1250 | 0.0261 |

Latest 25 by recipe family:

| Family | Terminal | Completed | Avg Quality | Avg Sharpe | Avg Fitness | Avg Turnover |
|---|---:|---:|---:|---:|---:|---:|
| `fundamental_quality` | 7 | 7 | 0.2530 | 0.6057 | 0.3086 | 0.0263 |
| `value_vs_growth` | 6 | 6 | 0.1674 | 0.3483 | 0.1533 | 0.0832 |
| `revision_surprise` | 13 | 13 | 0.1525 | 0.3200 | 0.1577 | 0.0321 |
| `accrual_vs_cashflow` | 6 | 5 | -0.0236 | -0.1060 | -0.0440 | 0.0388 |

Interpretation:
- Previous report favored `revision_surprise`; latest results shift the strongest evidence toward `fundamental_quality`, especially `quality` and `low_turnover`.
- `value_vs_growth|balanced` is viable and should stay in the mix.
- `accrual_vs_cashflow|quality` is currently harmful; `accrual_vs_cashflow|balanced` is weak but not catastrophically bad.

## Motif and Structure

Post-report motif quality:

| Motif | Terminal | Completed | Avg Quality | Avg Sharpe | Avg Fitness | Avg Turnover |
|---|---:|---:|---:|---:|---:|---:|
| `recipe_fundamental_quality` | 7 | 7 | 0.2530 | 0.6057 | 0.3086 | 0.0263 |
| `recipe_value_vs_growth` | 6 | 6 | 0.1674 | 0.3483 | 0.1533 | 0.0832 |
| `recipe_revision_surprise` | 13 | 13 | 0.1525 | 0.3200 | 0.1577 | 0.0321 |
| `momentum` | 24 | 23 | 0.0835 | 0.2257 | 0.1078 | 0.0536 |
| `spread` | 20 | 18 | 0.0804 | 0.2150 | 0.1183 | 0.0580 |
| `group_relative_signal` | 28 | 27 | 0.0635 | 0.2596 | 0.1070 | 0.0517 |
| `regime_conditioned_signal` | 27 | 26 | 0.0573 | 0.2265 | 0.0958 | 0.0604 |
| `recipe_accrual_vs_cashflow` | 6 | 5 | -0.0236 | -0.1060 | -0.0440 | 0.0388 |
| `price_volume_divergence` | 11 | 10 | -0.0600 | -0.0120 | 0.0089 | 0.1953 |
| `mean_reversion` | 9 | 9 | -0.1469 | -0.3089 | -0.1033 | 0.1651 |
| `residualized_signal` | 10 | 10 | -0.1628 | -0.1020 | -0.0420 | 0.0355 |

Structural notes:
- `analyst`-only field families are currently best post-report: `34` terminal, `33` completed, avg quality `0.1625`.
- `rank|ts_mean` is the strongest repeated operator path: latest 25 avg quality `0.1900`, avg Sharpe `0.4392`, avg fitness `0.2262`, avg turnover `0.0217`.
- Long-horizon alphas are better than short/medium recently: post-report long horizon avg quality `0.0549`; medium `0.0087`; short `-0.0061`.

## Top Recent Completed Alphas

Top post-report completed candidates by quality score:

| Round | Candidate | Source | Bucket/Motif | Quality | Sharpe | Fitness | Turnover | Main issue |
|---:|---|---|---|---:|---:|---:|---:|---|
| 12581 | `beadce3a8151b85d` | fresh | `momentum` | 0.4368 | 1.02 | 0.87 | 0.0194 | `LOW_SHARPE`, `LOW_FITNESS`, sub-universe, 2Y |
| 12582 | `bcb55dbe76fccf9f` | `quality_polish` | `quality_polish` | 0.4342 | 1.01 | 0.87 | 0.0219 | `LOW_SHARPE`, `LOW_FITNESS`, sub-universe |
| 12568 | `3d382f96a759229a` | fresh | `group_relative_signal` | 0.3819 | 1.01 | 0.61 | 0.0278 | Check metadata incomplete/empty |
| 12603 | `1c7d113efc2532c7` | `recipe_guided` | `fundamental_quality|quality` | 0.3801 | 1.09 | 0.53 | 0.0220 | `LOW_SHARPE`, `LOW_FITNESS`, sub-universe, 2Y |
| 12598 | `0e45770cec77efe3` | fresh | `spread` | 0.3699 | 0.88 | 0.66 | 0.0658 | `LOW_SHARPE`, `LOW_FITNESS`, sub-universe, 2Y |

The best realized candidates are close but still fail on absolute Sharpe/Fitness thresholds. The current search is finding better shape, not yet submission-clean alpha.

## BRAIN Checks and Rejections

Across completed BRAIN results:

| Check result | Count |
|---|---:|
| `LOW_SHARPE = FAIL` | 2,174 |
| `LOW_FITNESS = FAIL` | 2,161 |
| `LOW_2Y_SHARPE = FAIL` | 1,354 |
| `LOW_SUB_UNIVERSE_SHARPE = FAIL` | 840 |
| `IS_LADDER_SHARPE = FAIL` | 782 |
| `CONCENTRATED_WEIGHT = FAIL` | 746 |
| `LOW_TURNOVER = FAIL` | 99 |
| `HIGH_TURNOVER = FAIL` | 82 |

Top rejection/status reasons:

| Reason | Count |
|---|---:|
| `poll_timeout` | 934 |
| `poll_timeout_after_downtime` | 744 |
| `poll_timeout_live` | 77 |
| Reversion component warning/rejection | 185 |
| `Operator group_neutralize does not support event inputs` | 102 |
| Invalid or unknown fields, especially malformed analyst field names | dozens |

Log-level service pressure:

| Event/status | Count |
|---|---:|
| `service_tick_completed` | 3,889 |
| `batch_polled` | 1,211 |
| `batch_submitted` | 1,160 |
| `queue_prepare_deferred` | 821 |
| `waiting_persona_confirmation` status | 1,254 |
| Concurrent simulation limit errors | 270 |

Interpretation:
- Timeout is still the largest throughput loss.
- Persona confirmation is now a direct uptime limiter.
- Quality failures are not mostly turnover; they are mostly low Sharpe/Fitness and robustness checks.
- There are still avoidable expression validity issues: event-input misuse, invalid field names, and unit mismatch.

## Mutation Outcomes

Mutation outcomes are weak overall:

| Mode | N | Avg outcome delta | Positive delta rate | Avg quality delta | Positive quality delta rate |
|---|---:|---:|---:|---:|---:|
| All mutations | 801 | -0.5798 | 1.5% | -0.0184 | 0.0% |
| `recipe_guided` | 10 | -0.3123 | 0.0% | -0.3931 | 0.0% |
| `quality_polish` | 5 | -0.0533 | 20.0% | -0.0836 | 0.0% |
| `exploit_local` | 7 | -0.4439 | 0.0% | -0.4497 | 0.0% |
| `repair` | 1 | -0.8908 | 0.0% | -0.8557 | 0.0% |
| `crossover` | 1 | -0.0979 | 0.0% | -0.1781 | 0.0% |

Interpretation:
- Mutation is not currently improving children over parents.
- It should not receive more budget until mutation policy learns from winners or is narrowed to small, evidence-backed transforms.
- `quality_polish` is least bad among mutations, but still negative on quality delta.

## Recommendations

For the next run window:

1. Increase `recipe_guided`, but concentrate it.
   - Prioritize `fundamental_quality|fundamental|quality`.
   - Prioritize `fundamental_quality|fundamental|low_turnover`.
   - Keep `revision_surprise|balanced` and `revision_surprise|low_turnover` as steady secondary buckets.
   - Keep `value_vs_growth|balanced` in exploration.

2. Reduce or floor weak buckets.
   - Floor `accrual_vs_cashflow|fundamental|quality`.
   - Keep `accrual_vs_cashflow|balanced` small only for exploration.
   - Do not expand `price_volume_divergence`, `mean_reversion`, or `residualized_signal` until they show better recent BRAIN quality.

3. Keep `quality_polish`, but fix production volume before increasing budget.
   - Quality is excellent on the tiny sample.
   - Latest 10 rounds generated `0`, so budget alone is not converting into output.
   - Investigate transform cooldown, parent reuse limits, and signature blocking.

4. Tighten BRAIN-readiness filters around Sharpe/Fitness robustness.
   - Current top alphas still fail `LOW_SHARPE` and `LOW_FITNESS`.
   - Selection should penalize candidates that are likely to score under Sharpe/Fitness thresholds even if quality score is positive.
   - Add stronger 2Y/sub-universe proxies if available from BRAIN results.

5. Fix service reliability before interpreting timeout-heavy windows too strongly.
   - Clean up stale `running` jobs past timeout deadline.
   - Reduce Persona confirmation downtime.
   - Back off more cleanly on concurrent simulation limit.

6. Keep turnover repair off unless it has a real trigger.
   - Turnover is not the dominant failure mode.
   - Recent good recipe buckets already have low turnover.

## Suggested Next Budget Mix

Pragmatic next-window source mix:

| Source | Suggested candidates | Reason |
|---|---:|---|
| `recipe_guided` | 28-34 | Best scalable realized quality |
| `fresh` | 34-40 | Still needed for outliers and diversity |
| `quality_polish` | 16-22 | High quality, but output blocked/exhausted |
| mutation/repair | minimum | Negative child deltas |

Suggested recipe split inside `recipe_guided`:

| Bucket | Suggested share |
|---|---:|
| `fundamental_quality|fundamental|quality` | 25-30% |
| `fundamental_quality|fundamental|low_turnover` | 20-25% |
| `revision_surprise|fundamental|balanced` | 15-20% |
| `revision_surprise|fundamental|low_turnover` | 10-15% |
| `value_vs_growth|fundamental|balanced` | 10-15% |
| Other buckets | floor only |

Bottom line: the alpha factory is improving, especially after round `12567`, but it is not yet producing submission-clean alphas. The best path is narrower recipe-guided exploitation around analyst/fundamental-quality long-horizon `rank|ts_mean` structures, while fixing BRAIN service uptime and adding stronger pre-submit Sharpe/Fitness robustness filters.
