# FindingAlpha Knowledge Base

Bo docs nay bien `FindingAlpha.pdf` thanh knowledge base tra cuu nhanh cho `WQ Tool`.
Muc tieu khong phai la tom tat sach theo kieu hoc thuat, ma la paraphrase y chinh roi map truc tiep vao van de thuong gap trong repo: quality, turnover, correlation, overfitting, automated search, va fundamental recipe design.

Neu can doc nhanh, bat dau tu:

- [07_turnover.md](./07_turnover.md)
- [08_alpha_correlation.md](./08_alpha_correlation.md)
- [09_backtest_signal_or_overfitting.md](./09_backtest_signal_or_overfitting.md)
- [11_triple_axis_plan.md](./11_triple_axis_plan.md)
- [12_robustness.md](./12_robustness.md)
- [15_automated_search.md](./15_automated_search.md)
- [20_fundamental_alpha_research.md](./20_fundamental_alpha_research.md)
- [31_websim.md](./31_websim.md)

## Start Here

| Problem | Start with | Then read |
| --- | --- | --- |
| Quality / Sharpe / Fitness dang dung lai | [15 Automated Search](./15_automated_search.md) | [12 Robustness](./12_robustness.md), [11 Triple-Axis Plan](./11_triple_axis_plan.md) |
| Timeout cao lam kho doc quality that | [09 Backtest - Signal or Overfitting?](./09_backtest_signal_or_overfitting.md) | [31 WebSim](./31_websim.md) |
| Turnover cao hoac can low-turnover variants | [07 Turnover](./07_turnover.md) | [12 Robustness](./12_robustness.md) |
| Duplicate / near-duplicate / crowding tang | [08 Alpha Correlation](./08_alpha_correlation.md) | [11 Triple-Axis Plan](./11_triple_axis_plan.md) |
| Recipe generation cho fundamental chua ra alpha dep | [20 Fundamental Analysis and Alpha Research](./20_fundamental_alpha_research.md) | [15 Automated Search](./15_automated_search.md) |
| Muon mo rong search space nhung khong bi loang | [11 Triple-Axis Plan](./11_triple_axis_plan.md) | [15 Automated Search](./15_automated_search.md) |
| Muon robust hon truoc outlier va instability | [12 Robustness](./12_robustness.md) | [07 Turnover](./07_turnover.md) |
| Can hieu mindset WebSim/BRAIN workflow | [31 WebSim](./31_websim.md) | [09 Backtest - Signal or Overfitting?](./09_backtest_signal_or_overfitting.md) |

## Topic Map

| Topic | Main file | Supporting files |
| --- | --- | --- |
| quality improvement | [15 Automated Search](./15_automated_search.md) | [12 Robustness](./12_robustness.md), [11 Triple-Axis Plan](./11_triple_axis_plan.md) |
| turnover reduction | [07 Turnover](./07_turnover.md) | [12 Robustness](./12_robustness.md) |
| correlation / crowding | [08 Alpha Correlation](./08_alpha_correlation.md) | [11 Triple-Axis Plan](./11_triple_axis_plan.md) |
| fundamental recipes | [20 Fundamental Analysis and Alpha Research](./20_fundamental_alpha_research.md) | [11 Triple-Axis Plan](./11_triple_axis_plan.md), [15 Automated Search](./15_automated_search.md) |
| automated search | [15 Automated Search](./15_automated_search.md) | [09 Backtest - Signal or Overfitting?](./09_backtest_signal_or_overfitting.md) |
| overfitting diagnosis | [09 Backtest - Signal or Overfitting?](./09_backtest_signal_or_overfitting.md) | [12 Robustness](./12_robustness.md), [15 Automated Search](./15_automated_search.md) |

## Problem -> File

| Problem | File |
| --- | --- |
| turnover spikes / tradability | [07_turnover.md](./07_turnover.md) |
| duplicate / family concentration / crowding | [08_alpha_correlation.md](./08_alpha_correlation.md) |
| sample nho / noisy survivors / selection bias | [09_backtest_signal_or_overfitting.md](./09_backtest_signal_or_overfitting.md) |
| search space bi lech / thieu coverage | [11_triple_axis_plan.md](./11_triple_axis_plan.md) |
| outlier sensitivity / wrapper choice | [12_robustness.md](./12_robustness.md) |
| batch yield / search algorithm quality | [15_automated_search.md](./15_automated_search.md) |
| fundamental recipe design | [20_fundamental_alpha_research.md](./20_fundamental_alpha_research.md) |
| WebSim-like workflow / idea sourcing / operational usage | [31_websim.md](./31_websim.md) |

## Chapter -> File

| Chapter | File |
| --- | --- |
| 7 Turnover | [07_turnover.md](./07_turnover.md) |
| 8 Alpha Correlation | [08_alpha_correlation.md](./08_alpha_correlation.md) |
| 9 Backtest - Signal or Overfitting? | [09_backtest_signal_or_overfitting.md](./09_backtest_signal_or_overfitting.md) |
| 11 The Triple-Axis Plan | [11_triple_axis_plan.md](./11_triple_axis_plan.md) |
| 12 Techniques for Improving the Robustness of Alphas | [12_robustness.md](./12_robustness.md) |
| 15 Alphas from Automated Search | [15_automated_search.md](./15_automated_search.md) |
| 20 Fundamental Analysis and Alpha Research | [20_fundamental_alpha_research.md](./20_fundamental_alpha_research.md) |
| 31 Introduction to WebSim | [31_websim.md](./31_websim.md) |

## Repo Area -> File

| Repo area / concept | File |
| --- | --- |
| `services/quality_polisher.py` | [07_turnover.md](./07_turnover.md), [15_automated_search.md](./15_automated_search.md) |
| `services/recipe_guided_generator.py` | [11_triple_axis_plan.md](./11_triple_axis_plan.md), [20_fundamental_alpha_research.md](./20_fundamental_alpha_research.md) |
| `services/selection_service.py` | [08_alpha_correlation.md](./08_alpha_correlation.md), [09_backtest_signal_or_overfitting.md](./09_backtest_signal_or_overfitting.md) |
| `services/brain_batch_service.py` | [15_automated_search.md](./15_automated_search.md), [11_triple_axis_plan.md](./11_triple_axis_plan.md) |
| `services/kpi_report_service.py` | [08_alpha_correlation.md](./08_alpha_correlation.md), [09_backtest_signal_or_overfitting.md](./09_backtest_signal_or_overfitting.md) |
| `config/brain_full.yaml` tuning | [07_turnover.md](./07_turnover.md), [11_triple_axis_plan.md](./11_triple_axis_plan.md), [15_automated_search.md](./15_automated_search.md) |

## Current Repo Hooks

Nhung hook hien da ton tai trong repo va nen duoc doc cung voi playbooks:

- `quality_polish`: local source de exploit parent tot va generate controlled variants
- `recipe_guided`: bucket-based source de organize fundamental-heavy search space
- `search_bucket`: mapping `recipe_family|dataset_family|objective_profile`
- `family_correlation_proxy_penalty`: pre-sim penalty cho batch/family concentration
- `turnover_repair`: repair lane de tao smoothing variants khi turnover qua cao

## How To Use This Folder

- Gap issue van hanh hoac quality: doc bang `Start Here` truoc
- Muon brainstorm patch moi: doc 1 file playbook, sau do moi map sang code
- Muon bao ve minh khoi ket luan sai: doc [09 Backtest - Signal or Overfitting?](./09_backtest_signal_or_overfitting.md) truoc khi phan xet tu sample nho
- Muon mo rong search: doc [11 Triple-Axis Plan](./11_triple_axis_plan.md) truoc khi them mode/bucket moi

## Phase 2 Candidates

Neu V1 co gia tri, cac chuong nen mo rong tiep:

- `4 Alpha Design`
- `5 How to Develop an Alpha: A Case Study`
- `6 Data and Alpha Design`
- `10 Controlling Biases`
- `13 Alpha and Risk Factors`
- `14 Risk and Drawdowns`
- `16 Machine Learning in Alpha Research`
- `17 Thinking in Algorithms`
- `18 Equity Price and Volume`
- `19 Financial Statement Analysis`
- `21 Introduction to Momentum Alphas`
- `25 Event-Driven Investing`
- `26 Intraday Data in Alpha Research`

## Notes

- V1 uu tien usefulness cho repo hien tai, khong co gang bao phu toan bo cuon sach.
- Noi dung duoc paraphrase tu `FindingAlpha.pdf`, khong copy lai chapter text theo dang note hoc thuat.
- Neu co patch moi lam thay doi khung generator/selection/reporting, cap nhat cac `Apply In WQ Tool` sections truoc.
