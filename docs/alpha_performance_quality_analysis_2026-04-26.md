# Phân tích hiệu suất và chất lượng alpha

Run: `8c1f78cc6618`  
Cơ sở dữ liệu: `dev_wq_tool.sqlite3`  
Log: `progress_logs/8c1f78cc6618*.jsonl`  
Ngày phân tích: `2026-04-26`  
Cửa sổ phân tích chính: vòng `12606-12644`  

Ghi chú:
- Timestamp trong DB/log là UTC.
- Báo cáo trước kết thúc ở vòng `12605`.
- Vòng mới nhất trong DB là `12644`.
- Vòng `12644` vẫn còn `8` BRAIN job đang pending.
- Số lượng BRAIN check có tính cả check `PENDING` cho correlation/submission, nên "không có FAIL check" không đồng nghĩa với alpha đã được accept/submit thành công.

## Tóm tắt điều hành

Khoảng 20 giờ chạy vừa rồi đã đủ dữ liệu để cập nhật đánh giá:

- Throughput cải thiện rõ.
- `recipe_guided` là source tốt nhất hiện tại.
- Chất lượng tổng thể giảm so với cửa sổ báo cáo trước vì `fresh` tạo ra nhiều kết quả yếu và nhiều timeout.
- `quality_polish` đang gần như không hoạt động: vẫn được cấp budget nhưng generate `0`.
- Tool đang có dấu hiệu thích nghi, nhưng bài toán chất lượng alpha chưa được giải quyết: không có alpha completed nào trong cửa sổ mới vượt qua toàn bộ BRAIN check trả về mà không có FAIL.

| Cửa sổ | Số vòng | BRAIN job đã kết thúc | Completed | Tỷ lệ completed | Sharpe TB | Fitness TB | Quality TB | Tỷ lệ quality dương |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline `12545-12555` | 11 | 115 | 28 | 24.3% | 0.1014 | 0.0544 | 0.0222 | 64.3% |
| V2 start `12556-12566` | 11 | 111 | 40 | 36.0% | 0.1387 | 0.1070 | 0.0352 | 62.5% |
| Báo cáo trước `12567-12605` | 39 | 418 | 187 | 44.7% | 0.1767 | 0.0891 | 0.0449 | 65.2% |
| 20 giờ mới `12606-12644` | 39 | 438 | 224 | 51.1% | 0.0866 | 0.0299 | -0.0100 | 56.3% |
| 25 vòng mới nhất `12620-12644` | 25 | 277 | 136 | 49.1% | 0.0744 | 0.0269 | -0.0172 | 53.7% |
| 10 vòng mới nhất `12635-12644` | 10 | 105 | 43 | 41.0% | 0.0744 | 0.0176 | -0.0281 | 51.2% |

Diễn giải:
- Tỷ lệ completed tăng từ `44.7%` lên `51.1%` trong cửa sổ 20 giờ mới.
- Tỷ lệ validate và submit cũng cải thiện.
- Nhưng Sharpe, fitness và quality thực tế đều giảm.
- 10 vòng mới nhất đặc biệt yếu và vẫn còn `8` job đang running, nên kết quả đoạn cuối chưa hoàn toàn settled.

## Trạng thái service hiện tại

Runtime mới nhất của service:

| Trường | Giá trị |
|---|---|
| Trạng thái service | `service_stopped_pending` |
| Vòng mới nhất | `12644` |
| Batch đang active | `brain-8c1f78cc-r12644-49f9903b` |
| Job pending | `8` |
| Heartbeat cuối | `2026-04-26T04:23:10+00:00` |
| Success cuối | `2026-04-26T04:23:10+00:00` |
| Error cuối | `None` |
| Pending cap đã học | `8` |
| Số lần chạm limit quan sát được | `311` |

Service ở trạng thái tốt hơn báo cáo trước: tại heartbeat cuối không còn Persona error active, và cơ chế học pending cap đang hoạt động. Tuy vậy, service vẫn dừng khi còn `8` job pending.

## Throughput pipeline

| Cửa sổ | Generated | Validated | Submitted | Completed theo aggregate vòng | Tỷ lệ validate | Tỷ lệ submit |
|---|---:|---:|---:|---:|---:|---:|
| Báo cáo trước `12567-12605` | 3,900 | 2,350 | 418 | 418 | 60.3% | 10.7% |
| 20 giờ mới `12606-12644` | 3,900 | 2,471 | 446 | 438 | 63.4% | 11.4% |
| 25 vòng mới nhất `12620-12644` | 2,500 | 1,583 | 285 | 277 | 63.3% | 11.4% |
| 10 vòng mới nhất `12635-12644` | 1,000 | 638 | 113 | 105 | 63.8% | 11.3% |

Trạng thái submission/BRAIN trong `12606-12644`:

| Trạng thái | Số lượng |
|---|---:|
| Submission completed | 224 |
| Submission timeout | 214 |
| Submission running | 8 |
| BRAIN result completed | 224 |
| BRAIN result timeout | 214 |

Diễn giải:
- Hiệu suất pipeline local đã cải thiện.
- BRAIN hoàn tất nhiều kết quả hơn so với trước.
- Timeout vẫn lớn nhưng đã bớt áp đảo hơn.
- Vấn đề chất lượng còn lại không chỉ do timeout; nhóm alpha completed cũng yếu hơn trung bình.

## Hiệu suất theo source

Chất lượng BRAIN thực tế trong 20 giờ mới:

| Source | Job kết thúc | Completed | Tỷ lệ completed | Sharpe TB | Fitness TB | Quality TB | Quality median | Tỷ lệ quality dương | Turnover TB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `recipe_guided` | 76 | 76 | 100.0% | 0.2033 | 0.0955 | 0.0882 | 0.1409 | 75.0% | 0.0524 |
| `fresh` | 362 | 148 | 40.9% | 0.0267 | -0.0032 | -0.0604 | -0.0763 | 46.6% | 0.0743 |

Chất lượng BRAIN thực tế trong báo cáo trước:

| Source | Job kết thúc | Completed | Tỷ lệ completed | Sharpe TB | Fitness TB | Quality TB | Tỷ lệ quality dương |
|---|---:|---:|---:|---:|---:|---:|---:|
| `quality_polish` | 2 | 2 | 100.0% | 0.8550 | 0.6450 | 0.3623 | 100.0% |
| `recipe_guided` | 32 | 31 | 96.9% | 0.3213 | 0.1584 | 0.1496 | 90.3% |
| `fresh` | 383 | 154 | 40.2% | 0.1388 | 0.0678 | 0.0197 | 59.7% |

Diễn giải:
- `recipe_guided` vẫn là nguồn thắng rõ ràng.
- Chất lượng `recipe_guided` giảm so với báo cáo trước, nhưng vẫn giữ được mức dương.
- Chất lượng `fresh` sụp xuống net âm.
- Không thể đánh giá `quality_polish` trong cửa sổ mới vì source này không tạo ra candidate nào.

## Budget sinh alpha và conversion

Telemetry sinh alpha trong 20 giờ mới:

| Source | Budget được cấp | Generated | Selected |
|---|---:|---:|---:|
| `fresh` | 1,430 | 2,619 | 258 |
| `quality_polish` | 1,015 | 0 | 0 |
| `recipe_guided` | 780 | 696 | 158 |
| `mutation` | n/a | n/a | 35 |

Conversion:

| Source | Selected / Generated |
|---|---:|
| `recipe_guided` | 22.7% |
| `fresh` | 9.9% |
| `quality_polish` | 0.0% |

Phát hiện vận hành quan trọng:
- `quality_polish_attempt_count = 46`
- `quality_polish_success_count = 0`
- `quality_polish_generated = 0`
- `quality_polish_selected = 0`

Điều này nghĩa là `quality_polish` không chỉ underperform; nó đang bị block hoặc cạn nguồn. Budget cấp cho source này hiện đang bị lãng phí.

Áp lực sinh candidate của recipe:

| Metric | 20 giờ mới |
|---|---:|
| `recipe_guided_attempt_count` | 1,485 |
| `recipe_guided_success_count` | 696 |
| `recipe_guided_duplicate_retry_count` | 789 |
| `recipe_guided_unique_draft_count` | 4,576 |
| `recipe_guided_spilled_to_fresh` | 84 |

Diễn giải:
- Recipe generator đang làm nhiều việc hơn và tạo ra nhiều candidate hơn hẳn.
- Duplicate retry và spill-to-fresh cao, cho thấy một số bucket đang bị cạn, lặp hoặc thiếu biến thể tốt.
- Field rotation giúp tăng volume, nhưng lợi thế chất lượng không giữ được ở cùng mức.

## Chất lượng theo recipe bucket

Kết quả bucket trong 20 giờ mới:

| Bucket | Completed | Quality TB | Quality median | Sharpe TB | Fitness TB | Turnover TB | Tỷ lệ quality dương |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fundamental_quality|fundamental|quality` | 1 | 0.2433 | 0.2433 | 0.6100 | 0.2800 | 0.0269 | 100.0% |
| `fundamental_quality|fundamental|low_turnover` | 2 | 0.1659 | 0.1659 | 0.3150 | 0.1100 | 0.0097 | 100.0% |
| `value_vs_growth|fundamental|balanced` | 10 | 0.1529 | 0.1539 | 0.3570 | 0.1750 | 0.0796 | 90.0% |
| `fundamental_quality|fundamental|balanced` | 28 | 0.1191 | 0.1575 | 0.2700 | 0.1300 | 0.0511 | 82.1% |
| `revision_surprise|fundamental|low_turnover` | 5 | 0.1063 | 0.1605 | 0.2520 | 0.0840 | 0.0297 | 80.0% |
| `value_vs_growth|fundamental|quality` | 8 | 0.0846 | 0.1200 | 0.2162 | 0.0800 | 0.0237 | 75.0% |
| `revision_surprise|fundamental|balanced` | 11 | 0.0405 | 0.1222 | 0.0636 | 0.0218 | 0.0694 | 72.7% |
| `revision_surprise|fundamental|quality` | 6 | -0.0240 | -0.0042 | -0.0417 | 0.0050 | 0.0711 | 50.0% |
| `accrual_vs_cashflow|fundamental|quality` | 3 | -0.0368 | 0.0000 | -0.0833 | -0.0650 | 0.0493 | 33.3% |
| `value_vs_growth|fundamental|low_turnover` | 1 | -0.1315 | -0.1315 | -0.1100 | -0.0200 | 0.0366 | 0.0% |

Góc nhìn theo family:

| Family | Completed | Quality TB | Sharpe TB | Fitness TB | Tỷ lệ quality dương |
|---|---:|---:|---:|---:|---:|
| `fundamental_quality` | 31 | 0.1261 | 0.2839 | 0.1335 | 83.9% |
| `value_vs_growth` | 19 | 0.1092 | 0.2732 | 0.1247 | 78.9% |
| `revision_surprise` | 22 | 0.0379 | 0.0777 | 0.0314 | 68.2% |
| `accrual_vs_cashflow` | 4 | -0.0276 | -0.0625 | -0.0650 | 25.0% |

Diễn giải:
- `fundamental_quality` vẫn là family mạnh nhất xét theo cả số mẫu và chất lượng.
- `value_vs_growth|balanced` đang là bucket phụ rất đáng giữ.
- `revision_surprise` nguội đi rõ so với các cửa sổ trước.
- `accrual_vs_cashflow` vẫn yếu và nên bị hạ về floor.

## Áp lực selection theo bucket

Sinh và chọn recipe trong 20 giờ mới:

| Bucket | Generated | Selected | Selected / Generated |
|---|---:|---:|---:|
| `fundamental_quality|fundamental|balanced` | 70 | 49 | 70.0% |
| `revision_surprise|fundamental|balanced` | 73 | 29 | 39.7% |
| `value_vs_growth|fundamental|balanced` | 65 | 22 | 33.8% |
| `value_vs_growth|fundamental|quality` | 61 | 15 | 24.6% |
| `revision_surprise|fundamental|low_turnover` | 30 | 15 | 50.0% |
| `revision_surprise|fundamental|quality` | 62 | 12 | 19.4% |
| `fundamental_quality|fundamental|low_turnover` | 24 | 6 | 25.0% |
| `accrual_vs_cashflow|fundamental|quality` | 65 | 3 | 4.6% |
| `fundamental_quality|fundamental|quality` | 45 | 2 | 4.4% |
| `value_vs_growth|fundamental|low_turnover` | 57 | 2 | 3.5% |
| `accrual_vs_cashflow|fundamental|low_turnover` | 69 | 2 | 2.9% |
| `accrual_vs_cashflow|fundamental|balanced` | 75 | 1 | 1.3% |

Diễn giải:
- `fundamental_quality|balanced` overperform ở selection và vẫn dương trên BRAIN thực tế.
- `fundamental_quality|quality` có quality completed tốt nhưng conversion selection rất thấp; bucket này có thể cần thêm biến thể field/template tốt hơn.
- `accrual_vs_cashflow` tiêu tốn volume sinh alpha nhưng hiếm khi sống sót qua selection.
- Các bucket low-turnover có nhiều duplicate retry và output hạn chế; không nên kết luận là xấu ngay, nhưng cần nguồn field/pair mới.

## Motif và operator

Chất lượng motif trong 20 giờ mới:

| Motif | Completed | Quality TB | Sharpe TB | Fitness TB | Tỷ lệ quality dương |
|---|---:|---:|---:|---:|---:|
| `recipe_fundamental_quality` | 31 | 0.1261 | 0.2839 | 0.1335 | 83.9% |
| `recipe_value_vs_growth` | 19 | 0.1092 | 0.2732 | 0.1247 | 78.9% |
| `recipe_revision_surprise` | 22 | 0.0379 | 0.0777 | 0.0314 | 68.2% |
| `group_relative_signal` | 11 | 0.0363 | 0.2264 | 0.0982 | 54.5% |
| `volatility_adjusted_momentum` | 12 | -0.0017 | 0.1467 | 0.0483 | 58.3% |
| `regime_conditioned_signal` | 20 | -0.0225 | 0.0890 | 0.0400 | 60.0% |
| `recipe_accrual_vs_cashflow` | 4 | -0.0276 | -0.0625 | -0.0650 | 25.0% |
| `momentum` | 19 | -0.0581 | 0.0258 | -0.0583 | 47.4% |
| `spread` | 24 | -0.0706 | 0.0342 | 0.0104 | 41.7% |
| `mean_reversion` | 11 | -0.1523 | -0.2473 | -0.1109 | 27.3% |

Operator path có support mạnh nhất:

| Operator path | Completed | Quality TB | Sharpe TB | Fitness TB | Tỷ lệ quality dương |
|---|---:|---:|---:|---:|---:|
| `ts_mean|zscore` | 5 | 0.1643 | 0.4400 | 0.2120 | 80.0% |
| `rank|ts_mean` | 28 | 0.0960 | 0.2086 | 0.0936 | 78.6% |
| `binary:-|rank|ts_mean` | 14 | 0.0897 | 0.2521 | 0.1114 | 71.4% |
| `binary:-|rank` | 9 | 0.0786 | 0.1567 | 0.0971 | 66.7% |
| `rank` | 20 | 0.0616 | 0.1235 | 0.0575 | 75.0% |

Tín hiệu operator/motif yếu:
- `ts_corr`: quality TB `-0.0689`, turnover TB `0.1674`.
- `ts_rank|ts_mean`: quality TB `-0.2131`.
- `ts_rank|ts_std_dev`: quality TB `-0.2316`.
- `mean_reversion`: quality TB `-0.1523`.

Diễn giải:
- Các shape đơn giản và robust vẫn chiếm ưu thế: `rank`, `rank(ts_mean(...))`, `zscore(ts_mean(...))`.
- Shape kiểu correlation và reversion vẫn rủi ro.
- Vấn đề hiện tại không phải thiếu operator; vấn đề chính là ghép field và giữ kỷ luật shape robust.

## Top alpha completed trong cửa sổ mới

| Vòng | Candidate | Source | Bucket/Motif | Quality | Sharpe | Fitness | Turnover | Check fail chính |
|---:|---|---|---|---:|---:|---:|---:|---|
| 12616 | `68cd28ba5ad508c4` | `recipe_guided` | `value_vs_growth|balanced` | 0.3452 | 0.97 | 0.49 | 0.1054 | `LOW_SHARPE`, `LOW_FITNESS`, sub-universe |
| 12633 | `ec6c33c2d83f6b89` | `recipe_guided` | `fundamental_quality|balanced` | 0.3396 | 0.92 | 0.48 | 0.0671 | `LOW_SHARPE`, `LOW_FITNESS`, 2Y |
| 12630 | `865964698fa2e472` | fresh | `group_relative_signal` | 0.3259 | 0.84 | 0.46 | 0.0107 | `LOW_SHARPE`, `LOW_FITNESS` |
| 12617 | `0c86d6aceeabfa4a` | fresh | `group_relative_signal` | 0.3158 | 0.88 | 0.38 | 0.0179 | `LOW_SHARPE`, `LOW_FITNESS`, 2Y |
| 12636 | `8ec033db556b93dc` | `recipe_guided` | `fundamental_quality|balanced` | 0.3031 | 0.81 | 0.39 | 0.0606 | `LOW_SHARPE`, `LOW_FITNESS`, 2Y |

Các expression shape tốt nhất:
- `rank(ts_mean(field, d)) - rank(ts_mean(field, d))`
- `rank(field)`
- `zscore(ts_mean(field, d))`
- `rank(group_neutralize(ts_decay_linear(ts_std_dev(field, d), d), group))`

Ngay cả các alpha tốt nhất trong cửa sổ mới vẫn fail `LOW_SHARPE` và `LOW_FITNESS`. Chúng phù hợp để học parent/family hơn là winner có thể submit ngay.

## BRAIN check và rejection

Kết quả completed trong 20 giờ mới:

| Kết quả check | Số lượng |
|---|---:|
| `LOW_SHARPE = FAIL` | 224 |
| `LOW_FITNESS = FAIL` | 221 |
| `LOW_2Y_SHARPE = FAIL` | 141 |
| `LOW_SUB_UNIVERSE_SHARPE = FAIL` | 109 |
| `IS_LADDER_SHARPE = FAIL` | 76 |
| `CONCENTRATED_WEIGHT = FAIL` | 42 |
| `LOW_TURNOVER = FAIL` | 28 |

Lý do rejection/status trong 20 giờ mới:

| Lý do | Số lượng |
|---|---:|
| `poll_timeout_after_downtime` | 214 |
| Cảnh báo/rejection do reversion component | 28 |
| Các biến thể unit mismatch | 6 |

Diễn giải:
- Blocker chất lượng alpha chính vẫn là robustness của Sharpe/Fitness.
- Turnover không phải lỗi chi phối.
- Expression có reversion component vẫn là nguồn tạo đuôi xấu lặp lại.
- Timeout vẫn quan trọng, nhưng các result completed cũng đủ yếu để thấy cải thiện service đơn thuần sẽ không giải quyết chất lượng.

## Sức khỏe log service

Log sau cutoff của báo cáo trước (`2026-04-25T01:00:42+00:00`) cho thấy:

| Log item | Số lượng |
|---|---:|
| `service_tick_completed` | 1,303 |
| `batch_submitted` | 446 |
| `batch_polled` | 439 |
| `queue_prepare_deferred` | 447 |
| Status `waiting_persona_confirmation` | 642 |
| Status `cooldown` | 189 |
| Lỗi concurrent simulation limit | 119 |
| Lỗi Persona wait | 601 |

Diễn giải:
- Persona downtime vẫn còn, nhưng trạng thái cuối của service khỏe hơn hôm qua.
- Service đã học pending cap là `8`, đây là tín hiệu tốt.
- Vẫn có nhiều queue deferral và cooldown; throughput cải thiện dù còn ma sát vận hành.

## Kết quả mutation

Tất cả outcome mutation sau đợt chạy này vẫn âm:

| Mode | N | Outcome delta TB | Tỷ lệ delta dương | Quality delta TB | Tỷ lệ quality delta dương |
|---|---:|---:|---:|---:|---:|
| Tất cả mutation | 834 | -0.5755 | 1.4% | -0.0381 | 0.0% |
| `recipe_guided` | 24 | -0.2610 | 0.0% | -0.3543 | 0.0% |
| `quality_polish` | 5 | -0.0533 | 20.0% | -0.0836 | 0.0% |
| `exploit_local` | 10 | -0.4821 | 0.0% | -0.4673 | 0.0% |
| `crossover` | 3 | -0.5703 | 0.0% | -0.5219 | 0.0% |

Diễn giải:
- Mutation vẫn chưa phải cơ chế tự cải thiện đã được chứng minh.
- Nên giữ mutation ở budget thấp cho đến khi policy mutation có thể bảo toàn hoặc cải thiện quality.

## Chẩn đoán

Điểm đã cải thiện:
- Tỷ lệ BRAIN completed tăng lên `51.1%`.
- Tỷ lệ local validation và submit cải thiện.
- `recipe_guided` tạo volume cao hơn nhiều: `696` generated và `158` selected.
- `recipe_guided` có `100%` terminal completion trong cửa sổ mới.
- `fundamental_quality` và `value_vs_growth` vẫn là các family recipe hợp lệ.

Điểm xấu đi:
- Quality TB tổng thể giảm từ `0.0449` xuống `-0.0100`.
- Quality TB của 10 vòng mới nhất là `-0.0281`.
- Không có alpha mới nào đạt Sharpe >= `1.0`.
- `fresh` chuyển sang net âm.
- `quality_polish` không tạo candidate nào dù được cấp budget.
- `revision_surprise` nguội đi.
- `accrual_vs_cashflow` vẫn yếu.
- Mutation delta vẫn âm.

Nguyên nhân khả dĩ nhất:
1. Volume recipe tăng nhanh hơn khả năng kiểm soát chất lượng recipe.
2. Một số recipe bucket bị nặng duplicate/exhaustion.
3. Fresh exploration vẫn đưa vào nhiều alpha có đuôi âm.
4. `quality_polish` bị block, làm mất lane exploit chất lượng cao nhất ở báo cáo trước.
5. Selection vẫn chấp nhận candidate nhìn "hơi dương" nhưng còn quá xa ngưỡng Sharpe/Fitness của BRAIN.

## Khuyến nghị

### Có nên chạy tiếp không?

Có, nhưng không nên chạy tiếp với budget mix hiện tại mà không chỉnh.

Dữ liệu mới đủ để thấy tool vẫn tìm được cấu trúc hữu ích qua `recipe_guided`, nhưng cũng đủ để thấy mix hiện tại đang rò rỉ quá nhiều nhiễu từ `fresh` và các pattern reversion yếu.

### Budget theo source

Mix source đề xuất cho cửa sổ tiếp theo:

| Source | Khuyến nghị |
|---|---|
| `recipe_guided` | Tăng hoặc giữ cao; đây là source dương duy nhất có khả năng scale |
| `fresh` | Giảm so với realized volume hiện tại; chỉ giữ exploration floor |
| `quality_polish` | Fix ngay hoặc giảm budget vì output hiện tại là `0` |
| mutation/repair | Giữ tối thiểu; delta vẫn âm |

Target cụ thể:

| Source | Candidate budget đề xuất |
|---|---:|
| `recipe_guided` | 34-40 |
| `fresh` | 28-34 |
| `quality_polish` | 8-12 cho đến khi output được fix |
| mutation/repair | Chỉ giữ floor |

### Budget theo bucket

Tăng hoặc giữ:
- `fundamental_quality|fundamental|balanced`
- `fundamental_quality|fundamental|quality`
- `fundamental_quality|fundamental|low_turnover`
- `value_vs_growth|fundamental|balanced`
- `value_vs_growth|fundamental|quality`

Theo dõi nhưng không mở rộng quá mạnh:
- `revision_surprise|fundamental|balanced`
- `revision_surprise|fundamental|low_turnover`

Giảm về floor:
- `accrual_vs_cashflow|fundamental|balanced`
- `accrual_vs_cashflow|fundamental|quality`
- `accrual_vs_cashflow|fundamental|low_turnover`
- Motif `fresh` yếu: `mean_reversion`, `price_volume_divergence`, `residualized_signal`

### Fix generator/selection

Ưu tiên cao:
1. Fix điều kiện zero-output của `quality_polish`.
   - Kiểm tra transform cooldown, signature blocking, parent-transform reuse và parent pool bị cạn.
   - Allocation hiện tại cho `quality_polish` đang bị lãng phí.

2. Thêm risk pre-submit mạnh hơn cho các lỗi BRAIN.
   - Penalize các tín hiệu dự báo `LOW_SHARPE`, `LOW_FITNESS`, `LOW_2Y_SHARPE` và yếu ở sub-universe.
   - Quality score dương hiện tại chưa đủ.

3. Penalize pattern có reversion component sớm hơn.
   - Các đuôi xấu nhất đang lặp lại ở nhóm reversion-like.

4. Thêm throttling theo mức exhaustion của recipe bucket.
   - Nếu duplicate retry và spill-to-fresh cao, tạm giảm bucket đó hoặc rotate field mạnh hơn.

5. Giữ các shape đơn giản, robust.
   - Ưu tiên `rank`, `rank(ts_mean(...))`, `zscore(ts_mean(...))` và so sánh hai field có cấu trúc.
   - Không nên mở rộng operator set quá mạnh lúc này.

## Kết luận

Đợt chạy 20 giờ đã đủ dữ liệu để cập nhật thesis:

> Tool đang cải thiện về mặt vận hành và `recipe_guided` đang học được cấu trúc hữu ích, nhưng chất lượng alpha tổng thể không tiếp tục cải thiện. Bước tiếp theo không nên là "cứ chạy thêm y nguyên"; nên là "chạy tiếp sau khi giảm nhiễu từ fresh, fix output của quality_polish, và siết filter robustness cho Sharpe/Fitness".

Quyết định khuyến nghị:
- Tiếp tục chạy.
- Không coi 20 giờ mới nhất là một quality win.
- Coi nó là diagnostic về allocation: `recipe_guided` đang đúng hướng, `fresh` quá nhiễu, `quality_polish` đang bị block, và các FAIL theo threshold BRAIN cần trở thành penalty hạng nhất trong selection.
