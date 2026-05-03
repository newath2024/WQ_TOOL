# Current BRAIN Search Space For Claude

Generated at: 2026-05-01T11:34:03.894603+00:00 UTC
Config: `config/brain_full.yaml`
Database: `dev_wq_tool.sqlite3`
Current service run id used for recent usage: `8c1f78cc6618`
Service status snapshot: `running`, pid `64544`, pending `1`, updated `2026-05-01T11:30:57.836392+00:00`

Important note for Claude: use the `Currently eligible field registry` sections below, not the whole historical SQLite `field_catalog`. The historical table contains fields from old regions/configs; the current `brain_full.yaml` registry for USA/TOP3000 resolves to 834 fields.

## BRAIN Simulation Profile

| Setting | Value |
|---|---|
| region | `USA` |
| universe | `TOP3000` |
| delay | `1` |
| neutralization | `SUBINDUSTRY` |
| decay | `5` |
| truncation | `0.01` |
| pasteurization | `True` |
| unit_handling | `VERIFY` |
| nan_handling | `OFF` |
| timeout_seconds | `1000` |
| max_pending_jobs | `10` |
| batch_size | `10` |

## Operator Space

Configured allowed operators (30):

- `ts_delay`
- `ts_delta`
- `rank`
- `zscore`
- `ts_corr`
- `ts_covariance`
- `ts_decay_linear`
- `ts_rank`
- `ts_sum`
- `days_from_last_change`
- `ts_av_diff`
- `ts_scale`
- `ts_arg_max`
- `ts_arg_min`
- `quantile`
- `inverse`
- `reverse`
- `ts_count_nans`
- `min`
- `max`
- `ts_mean`
- `ts_std_dev`
- `ts_min`
- `ts_max`
- `sign`
- `abs`
- `log`
- `group_rank`
- `group_zscore`
- `group_neutralize`

Implicit arithmetic operators observed from generated expressions:

- `binary:*`
- `binary:+`
- `binary:-`
- `binary:/`
- `unary:-`

Lane-specific operator allowlists:

### `quality_polish` (21 operators)

`rank`, `zscore`, `quantile`, `ts_mean`, `ts_decay_linear`, `ts_std_dev`, `ts_rank`, `ts_sum`, `ts_scale`, `ts_arg_max`, `ts_arg_min`, `days_from_last_change`, `ts_av_diff`, `ts_count_nans`, `inverse`, `reverse`, `sign`, `abs`, `min`, `max`, `group_neutralize`

### `recipe_guided` (15 operators)

`rank`, `zscore`, `quantile`, `ts_mean`, `ts_std_dev`, `ts_scale`, `ts_av_diff`, `ts_arg_max`, `ts_arg_min`, `ts_count_nans`, `min`, `max`, `inverse`, `reverse`, `days_from_last_change`

### `fresh` (16 operators)

`rank`, `zscore`, `ts_mean`, `ts_decay_linear`, `ts_std_dev`, `ts_rank`, `ts_sum`, `ts_scale`, `days_from_last_change`, `ts_av_diff`, `ts_count_nans`, `sign`, `abs`, `min`, `max`, `group_neutralize`

Observed operators in generated candidates during last 8h (1141 alpha rows):

| Operator | Count |
|---|---:|
| `ts_decay_linear` | 515 |
| `ts_mean` | 509 |
| `ts_sum` | 331 |
| `ts_std_dev` | 316 |
| `rank` | 313 |
| `ts_rank` | 204 |
| `zscore` | 173 |
| `ts_corr` | 82 |
| `abs` | 80 |
| `binary:+` | 70 |
| `binary:/` | 68 |
| `ts_covariance` | 61 |
| `ts_delta` | 59 |
| `binary:-` | 57 |
| `binary:*` | 28 |
| `sign` | 26 |
| `unary:-` | 16 |
| `days_from_last_change` | 15 |
| `group_neutralize` | 8 |
| `ts_av_diff` | 7 |
| `ts_arg_max` | 6 |
| `ts_scale` | 5 |
| `ts_count_nans` | 5 |

Observed operators among submitted candidates after hotfix round >= 12754 (135 rows):

| Operator | Count |
|---|---:|
| `ts_mean` | 64 |
| `ts_decay_linear` | 60 |
| `rank` | 42 |
| `ts_sum` | 37 |
| `ts_std_dev` | 29 |
| `zscore` | 26 |
| `ts_rank` | 23 |
| `ts_delta` | 8 |
| `ts_corr` | 8 |
| `binary:-` | 5 |
| `sign` | 4 |
| `days_from_last_change` | 3 |
| `unary:-` | 3 |
| `abs` | 3 |
| `ts_scale` | 1 |
| `ts_av_diff` | 1 |

## Data Catalog Scope

Configured field catalog paths:

- `inputs/wq_snapshots/2026-03-29/Price Volumn`
- `inputs/wq_snapshots/2026-03-29/Fundamental`
- `inputs/wq_snapshots/2026-03-29/Earnings`
- `inputs/wq_snapshots/2026-03-29/Analyst`
- `inputs/wq_snapshots/2026-03-29/Model`

`generation.allowed_fields` is empty, so there is no explicit field whitelist. `allow_catalog_fields_without_runtime=True`.

Currently eligible field registry summary:

| Bucket | Count |
|---|---:|
| total fields | 834 |
| numeric/matrix fields usable as normal alpha inputs | 164 |
| group/vector fields usable as group keys | 670 |
| runtime sample fields | 14 |
| catalog-only fields | 820 |

By category:

| Category | Count |
|---|---:|
| `analyst` | 819 |
| `other` | 5 |
| `price` | 5 |
| `group` | 4 |
| `volume` | 1 |

By dataset:

| Dataset | Count |
|---|---:|
| `Fundamental Analyst Estimates` | 324 |
| `Analyst estimates & financial ratios` | 295 |
| `Integrated Broker Estimates` | 138 |
| `Analyst Trade Ideas` | 57 |
| `runtime` | 14 |
| `Analyst Investment insight Data` | 5 |
| `(blank)` | 1 |

Runtime sample fields merged into registry:

| Field | Operator type | Category |
|---|---|---|
| `country` | `group` | `group` |
| `industry` | `group` | `group` |
| `sector` | `group` | `group` |
| `subindustry` | `group` | `group` |
| `beta` | `matrix` | `other` |
| `close` | `matrix` | `price` |
| `high` | `matrix` | `price` |
| `liquidity` | `matrix` | `other` |
| `low` | `matrix` | `price` |
| `open` | `matrix` | `price` |
| `returns` | `matrix` | `price` |
| `size` | `matrix` | `other` |
| `volatility` | `matrix` | `other` |
| `volume` | `matrix` | `volume` |

Observed field usage in generated candidates during last 8h (1141 alpha rows):

Sources: {'fresh': 480, 'recipe_guided': 69, 'quality_polish': 352, 'mutation': 240}

| Field | Count |
|---|---:|
| `anl69_roe_expected_report_dt` | 136 |
| `anl69_cps_best_eeps_nxt_yr` | 115 |
| `anl69_ebit_best_crncy_iso` | 115 |
| `anl69_dps_best_cur_fiscal_qtr_period` | 106 |
| `anl69_cps_best_eeps_cur_yr` | 101 |
| `anl69_roa_best_cur_fiscal_qtr_period` | 98 |
| `country` | 65 |
| `anl39_curfyearend` | 62 |
| `anl69_eps_expected_report_dt` | 56 |
| `anl69_roe_best_cur_fiscal_year_period` | 56 |
| `sector` | 48 |
| `anl69_net_expected_report_dt` | 48 |
| `anl39_ttmepsincx` | 47 |
| `anl39_qepsinclxo` | 46 |
| `anl69_ndebt_best_crncy_iso` | 44 |
| `subindustry` | 43 |
| `anl39_agrosmgn2` | 41 |
| `industry` | 40 |
| `anl39_xlcxspemtt` | 39 |
| `anl39_ghcspemtt` | 39 |
| `anl69_ebit_best_eeps_cur_yr` | 39 |
| `anl39_curfperiodend` | 39 |
| `anl69_roa_best_cur_fiscal_year_period` | 39 |
| `anl39_qgrosmgn` | 38 |
| `anl39_rasv2_atotd2eq` | 38 |
| `anl69_eps_best_cur_fiscal_qtr_period` | 37 |
| `anl69_sales_expected_report_time` | 36 |
| `anl69_pe_best_cur_fiscal_year_period` | 36 |
| `anl69_dps_best_eeps_cur_yr` | 35 |
| `anl69_roa_best_crncy_iso` | 35 |
| `anl69_ebit_best_cur_fiscal_qtr_period` | 34 |
| `anl39_agrosmgn` | 33 |
| `anl39_aepsinclxo` | 33 |
| `anl69_ebit_best_eeps_nxt_yr` | 33 |
| `anl69_net_expected_report_time` | 33 |
| `anl69_roe_best_crncy_iso` | 33 |
| `anl69_pe_expected_report_dt` | 32 |
| `anl69_net_best_cur_fiscal_year_period` | 31 |
| `anl69_pe_best_eeps_cur_yr` | 31 |
| `anl69_roa_expected_report_dt` | 30 |
| `anl69_pe_expected_report_time` | 30 |
| `anl69_pe_best_eeps_nxt_yr` | 30 |
| `anl69_dps_expected_report_dt` | 30 |
| `anl39_epschngin` | 30 |
| `anl69_dps_expected_report_time` | 28 |
| `anl69_eps_best_eeps_cur_yr` | 27 |
| `anl69_pe_best_cur_fiscal_qtr_period` | 27 |
| `anl69_ebit_best_cur_fiscal_year_period` | 27 |
| `anl69_sales_best_cur_fiscal_qtr_period` | 26 |
| `anl69_sales_best_crncy_iso` | 26 |
| `anl69_roa_best_eeps_nxt_yr` | 26 |
| `anl69_net_best_cur_fiscal_qtr_period` | 26 |
| `anl69_ebit_expected_report_dt` | 25 |
| `anl69_sales_best_eeps_cur_yr` | 25 |
| `anl69_roa_best_eeps_cur_yr` | 25 |
| `anl69_dps_best_eeps_nxt_yr` | 25 |
| `anl69_ndebt_best_cur_fiscal_qtr_period` | 24 |
| `anl69_net_best_eeps_cur_yr` | 24 |
| `anl69_roe_best_eeps_nxt_yr` | 24 |
| `anl69_sales_best_cur_fiscal_year_period` | 23 |
| `anl69_roa_expected_report_time` | 23 |
| `anl69_ndebt_best_cur_fiscal_year_period` | 22 |
| `anl69_roe_best_cur_fiscal_qtr_period` | 21 |
| `anl39_qtanbvps` | 20 |
| `anl69_cps_expected_report_dt` | 18 |
| `anl69_ebit_expected_report_time` | 18 |
| `anl69_eps_best_cur_fiscal_year_period` | 16 |
| `anl69_ndebt_expected_report_dt` | 14 |
| `anl69_sales_expected_report_dt` | 14 |
| `anl39_qtotd2eq2` | 14 |
| `anl69_pe_best_crncy_iso` | 14 |
| `anl69_cps_expected_report_time` | 13 |
| `anl69_eps_best_eeps_nxt_yr` | 12 |
| `anl69_eps_best_crncy_iso` | 11 |
| `anl69_eps_expected_report_time` | 11 |
| `anl39_spvbq` | 10 |
| `anl69_roe_best_eeps_cur_yr` | 7 |
| `anl39_atanbvps` | 7 |
| `anl69_cps_best_crncy_iso` | 7 |
| `anl69_sales_best_eeps_nxt_yr` | 7 |
| `anl39_rygnhcspe` | 7 |
| `anl39_qtotd2eq` | 6 |
| `anl69_net_best_eeps_nxt_yr` | 5 |
| `anl39_xlcxspemtp` | 4 |
| `anl39_roxlcxspeq` | 4 |
| `high` | 4 |
| `anl39_ghcspea` | 4 |
| `anl69_dps_best_cur_fiscal_year_period` | 4 |
| `anl69_dps_best_crncy_iso` | 3 |
| `anl39_spvba` | 2 |
| `anl39_curperiodtype` | 2 |
| `low` | 2 |
| `open` | 2 |
| `anl69_cps_best_cur_fiscal_qtr_period` | 2 |
| `anl69_cps_best_cur_fiscal_year_period` | 2 |
| `close` | 2 |
| `anl39_grosmgn5yr` | 2 |
| `returns` | 2 |
| `anl39_ttmgrosmgn` | 2 |
| `anl69_net_best_crncy_iso` | 2 |
| `anl69_ndebt_best_eeps_cur_yr` | 2 |
| `anl69_roe_expected_report_time` | 1 |
| `anl39_ptmepsincx` | 1 |
| `volume` | 1 |
| `anl39_roxlcxspea` | 1 |
| `anl39_cursharesoutstanding` | 1 |

Observed field usage among submitted candidates after hotfix round >= 12754 (135 rows):

| Field | Count |
|---|---:|
| `anl69_cps_best_eeps_nxt_yr` | 11 |
| `anl69_pe_expected_report_dt` | 10 |
| `anl69_roe_best_crncy_iso` | 10 |
| `anl69_roe_expected_report_dt` | 9 |
| `anl69_roe_best_cur_fiscal_year_period` | 8 |
| `anl69_dps_expected_report_time` | 7 |
| `anl69_ebit_best_cur_fiscal_year_period` | 7 |
| `anl39_qepsinclxo` | 7 |
| `anl39_ttmepsincx` | 7 |
| `anl69_ebit_best_eeps_cur_yr` | 6 |
| `anl69_eps_best_cur_fiscal_qtr_period` | 6 |
| `anl39_agrosmgn2` | 6 |
| `anl69_net_expected_report_time` | 5 |
| `anl69_eps_expected_report_dt` | 5 |
| `anl69_dps_expected_report_dt` | 5 |
| `anl39_xlcxspemtt` | 5 |
| `anl69_ndebt_best_cur_fiscal_year_period` | 5 |
| `anl69_pe_expected_report_time` | 5 |
| `anl39_ghcspemtt` | 5 |
| `country` | 5 |
| `anl69_roa_best_cur_fiscal_year_period` | 5 |
| `anl69_net_expected_report_dt` | 5 |
| `anl39_curfperiodend` | 4 |
| `anl39_agrosmgn` | 4 |
| `anl39_qgrosmgn` | 4 |
| `anl69_ebit_best_eeps_nxt_yr` | 4 |
| `sector` | 4 |
| `anl69_roe_best_eeps_nxt_yr` | 4 |
| `anl69_ebit_best_crncy_iso` | 4 |
| `anl39_epschngin` | 4 |
| `anl69_ndebt_best_crncy_iso` | 4 |
| `anl69_eps_best_eeps_cur_yr` | 3 |
| `high` | 3 |
| `anl69_ebit_expected_report_time` | 3 |
| `anl39_rasv2_atotd2eq` | 3 |
| `anl69_roa_best_cur_fiscal_qtr_period` | 3 |
| `subindustry` | 3 |
| `anl69_sales_best_cur_fiscal_year_period` | 3 |
| `anl69_roa_expected_report_dt` | 3 |
| `anl39_atanbvps` | 3 |
| `anl69_roa_expected_report_time` | 3 |
| `anl69_net_best_cur_fiscal_year_period` | 3 |
| `anl69_roa_best_eeps_nxt_yr` | 3 |
| `anl69_sales_expected_report_time` | 3 |
| `anl69_sales_best_cur_fiscal_qtr_period` | 3 |
| `anl39_ghcspea` | 3 |
| `anl39_aepsinclxo` | 3 |
| `anl69_pe_best_eeps_cur_yr` | 3 |
| `anl39_qtanbvps` | 3 |
| `anl69_dps_best_eeps_nxt_yr` | 3 |
| `anl69_cps_expected_report_time` | 3 |
| `anl69_roa_best_crncy_iso` | 3 |
| `anl69_dps_best_eeps_cur_yr` | 3 |
| `low` | 2 |
| `anl39_xlcxspemtp` | 2 |
| `close` | 2 |
| `anl69_ndebt_best_cur_fiscal_qtr_period` | 2 |
| `anl69_eps_expected_report_time` | 2 |
| `anl69_ebit_best_cur_fiscal_qtr_period` | 2 |
| `anl69_sales_best_eeps_nxt_yr` | 2 |
| `anl69_cps_best_eeps_cur_yr` | 2 |
| `anl39_curfyearend` | 2 |
| `industry` | 2 |
| `anl69_eps_best_cur_fiscal_year_period` | 2 |
| `anl69_sales_best_eeps_cur_yr` | 2 |
| `anl69_sales_best_crncy_iso` | 2 |
| `anl69_cps_expected_report_dt` | 1 |
| `anl69_dps_best_cur_fiscal_qtr_period` | 1 |
| `anl39_roxlcxspea` | 1 |
| `open` | 1 |
| `volume` | 1 |
| `anl39_rygnhcspe` | 1 |
| `anl39_qtotd2eq2` | 1 |
| `anl39_grosmgn5yr` | 1 |
| `anl69_pe_best_crncy_iso` | 1 |
| `anl39_spvbq` | 1 |
| `anl39_ttmgrosmgn` | 1 |
| `anl39_roxlcxspeq` | 1 |
| `anl69_roe_best_cur_fiscal_qtr_period` | 1 |
| `returns` | 1 |
| `anl69_net_best_eeps_nxt_yr` | 1 |
| `anl69_sales_expected_report_dt` | 1 |
| `anl69_ndebt_expected_report_dt` | 1 |
| `anl69_pe_best_cur_fiscal_year_period` | 1 |
| `anl69_net_best_cur_fiscal_qtr_period` | 1 |
| `anl69_roa_best_eeps_cur_yr` | 1 |
| `anl69_eps_best_eeps_nxt_yr` | 1 |
| `anl69_ebit_expected_report_dt` | 1 |
| `anl69_pe_best_eeps_nxt_yr` | 1 |

## Currently Eligible Numeric/Matrix Fields

These are the main data fields the generator can use inside time-series/cross-sectional expressions.

| Field | Dataset | Category | Region | Universe | Coverage | Alpha usage | Score | Runtime |
|---|---|---|---|---|---:|---:|---:|---|
| `anl46_alphadecay` | `Analyst Investment insight Data` | `analyst` | `GLB` | `TOP3000` | 0.7301 | 8 | 0.606332 | `False` |
| `anl46_experts` | `Analyst Investment insight Data` | `analyst` | `GLB` | `TOP3000` | 0.6776 | 9 | 0.583020 | `False` |
| `anl46_indicator` | `Analyst Investment insight Data` | `analyst` | `GLB` | `TOP3000` | 0.7750 | 10 | 0.634379 | `False` |
| `anl46_performancepercentile` | `Analyst Investment insight Data` | `analyst` | `GLB` | `TOP3000` | 0.9386 | 21 | 0.735511 | `False` |
| `anl46_sentiment` | `Analyst Investment insight Data` | `analyst` | `GLB` | `TOP3000` | 0.7301 | 26 | 0.636973 | `False` |
| `anl39_aepsinclxo` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9478 | 1 | 0.673232 | `False` |
| `anl39_agrosmgn` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.8036 | 1 | 0.601132 | `False` |
| `anl39_agrosmgn2` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.8141 | 0 | 0.587050 | `False` |
| `anl39_atanbvps` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9503 | 1 | 0.674482 | `False` |
| `anl39_curfperiodend` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 45 | 0.786783 | `False` |
| `anl39_curfyearend` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 28 | 0.773916 | `False` |
| `anl39_curperiodtype` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 10 | 0.746879 | `False` |
| `anl39_cursharesoutstanding` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 33 | 0.778352 | `False` |
| `anl39_epschngin` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9401 | 0 | 0.650050 | `False` |
| `anl39_ghcspea` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9405 | 0 | 0.650250 | `False` |
| `anl39_ghcspemtt` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9181 | 0 | 0.639050 | `False` |
| `anl39_grosmgn5yr` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.7070 | 3 | 0.572165 | `False` |
| `anl39_ptmepsincx` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9412 | 0 | 0.650600 | `False` |
| `anl39_qepsinclxo` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9494 | 0 | 0.654700 | `False` |
| `anl39_qgrosmgn` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.8050 | 0 | 0.582500 | `False` |
| `anl39_qtanbvps` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9484 | 0 | 0.654200 | `False` |
| `anl39_qtotd2eq` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9297 | 4 | 0.689738 | `False` |
| `anl39_qtotd2eq2` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9072 | 1 | 0.652932 | `False` |
| `anl39_rasv2_atotd2eq` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9301 | 0 | 0.645050 | `False` |
| `anl39_roxlcxspea` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9478 | 0 | 0.653900 | `False` |
| `anl39_roxlcxspeq` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9494 | 0 | 0.654700 | `False` |
| `anl39_rygnhcspe` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9360 | 3 | 0.686665 | `False` |
| `anl39_spvba` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9503 | 0 | 0.655150 | `False` |
| `anl39_spvbq` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9484 | 0 | 0.654200 | `False` |
| `anl39_ttmepsincx` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9417 | 2 | 0.681491 | `False` |
| `anl39_ttmgrosmgn` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.8016 | 0 | 0.580800 | `False` |
| `anl39_xlcxspemtp` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9235 | 0 | 0.641750 | `False` |
| `anl39_xlcxspemtt` | `Analyst estimates & financial ratios` | `analyst` | `GLB` | `TOP3000` | 0.9417 | 0 | 0.650850 | `False` |
| `anl69_cps_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 56 | 0.792763 | `False` |
| `anl69_cps_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 29 | 0.770761 | `False` |
| `anl69_cps_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 9 | 0.544820 | `False` |
| `anl69_cps_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 8 | 0.741282 | `False` |
| `anl69_cps_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 31 | 0.776661 | `False` |
| `anl69_cps_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 36 | 0.780711 | `False` |
| `anl69_cps_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 10 | 0.746879 | `False` |
| `anl69_cps_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 8 | 0.741282 | `False` |
| `anl69_cpss_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 9 | 0.744220 | `False` |
| `anl69_cpss_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 12 | 0.747438 | `False` |
| `anl69_cpss_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 5 | 0.530573 | `False` |
| `anl69_cpss_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 12 | 0.751538 | `False` |
| `anl69_cpss_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 43 | 0.785543 | `False` |
| `anl69_cpss_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 193 | 0.826924 | `False` |
| `anl69_cpss_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 12 | 0.751538 | `False` |
| `anl69_cpss_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 11 | 0.749305 | `False` |
| `anl69_dps_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 122 | 0.814215 | `False` |
| `anl69_dps_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 42 | 0.780802 | `False` |
| `anl69_dps_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 5 | 0.530573 | `False` |
| `anl69_dps_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 13 | 0.753605 | `False` |
| `anl69_dps_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 10 | 0.746879 | `False` |
| `anl69_dps_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 22 | 0.767451 | `False` |
| `anl69_dps_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 12 | 0.751538 | `False` |
| `anl69_dps_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 2 | 0.710641 | `False` |
| `anl69_dpss_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 9 | 0.744220 | `False` |
| `anl69_dpss_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 41 | 0.780146 | `False` |
| `anl69_dpss_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 5 | 0.530573 | `False` |
| `anl69_dpss_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 10 | 0.746879 | `False` |
| `anl69_dpss_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 38 | 0.782179 | `False` |
| `anl69_dpss_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 43 | 0.785543 | `False` |
| `anl69_dpss_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 4 | 0.724888 | `False` |
| `anl69_dpss_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 4 | 0.724888 | `False` |
| `anl69_ebit_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 59 | 0.794194 | `False` |
| `anl69_ebit_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 8 | 0.737182 | `False` |
| `anl69_ebit_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 7 | 0.538597 | `False` |
| `anl69_ebit_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 3 | 0.718665 | `False` |
| `anl69_ebit_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 21 | 0.766211 | `False` |
| `anl69_ebit_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 19 | 0.763553 | `False` |
| `anl69_ebit_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 4 | 0.724888 | `False` |
| `anl69_ebit_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 5 | 0.729973 | `False` |
| `anl69_eps_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 3 | 0.718665 | `False` |
| `anl69_eps_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 13 | 0.749505 | `False` |
| `anl69_eps_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 0 | 0.480600 | `False` |
| `anl69_eps_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 15 | 0.757329 | `False` |
| `anl69_eps_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 18 | 0.762122 | `False` |
| `anl69_eps_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 20 | 0.764913 | `False` |
| `anl69_eps_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 0 | 0.680000 | `False` |
| `anl69_eps_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 2 | 0.710641 | `False` |
| `anl69_epss_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 1 | 0.699332 | `False` |
| `anl69_epss_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 14 | 0.751429 | `False` |
| `anl69_epss_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 1 | 0.499932 | `False` |
| `anl69_epss_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 7 | 0.737997 | `False` |
| `anl69_epss_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 10 | 0.746879 | `False` |
| `anl69_epss_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 17 | 0.760614 | `False` |
| `anl69_epss_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 7 | 0.737997 | `False` |
| `anl69_epss_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 8 | 0.741282 | `False` |
| `anl69_ndebt_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 1 | 0.699332 | `False` |
| `anl69_ndebt_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 8 | 0.737182 | `False` |
| `anl69_ndebt_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 3 | 0.519265 | `False` |
| `anl69_ndebt_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 6 | 0.734273 | `False` |
| `anl69_ndebt_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 6 | 0.734273 | `False` |
| `anl69_ndebt_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 17 | 0.760614 | `False` |
| `anl69_ndebt_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 25 | 0.770870 | `False` |
| `anl69_ndebt_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 3 | 0.718665 | `False` |
| `anl69_net_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 5 | 0.729973 | `False` |
| `anl69_net_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 9 | 0.740120 | `False` |
| `anl69_net_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 0 | 0.480600 | `False` |
| `anl69_net_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 4 | 0.724888 | `False` |
| `anl69_net_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 8 | 0.741282 | `False` |
| `anl69_net_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 12 | 0.751538 | `False` |
| `anl69_net_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 7 | 0.737997 | `False` |
| `anl69_net_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 4 | 0.724888 | `False` |
| `anl69_pe_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 5 | 0.729973 | `False` |
| `anl69_pe_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9935 | 15 | 0.754079 | `False` |
| `anl69_pe_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6029 | 0 | 0.481450 | `False` |
| `anl69_pe_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 6 | 0.734273 | `False` |
| `anl69_pe_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 21 | 0.766211 | `False` |
| `anl69_pe_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 22 | 0.767451 | `False` |
| `anl69_pe_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 1 | 0.699332 | `False` |
| `anl69_pe_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 4 | 0.724888 | `False` |
| `anl69_roa_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 18 | 0.762122 | `False` |
| `anl69_roa_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9935 | 33 | 0.775102 | `False` |
| `anl69_roa_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6029 | 0 | 0.481450 | `False` |
| `anl69_roa_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 10 | 0.746879 | `False` |
| `anl69_roa_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 14 | 0.755529 | `False` |
| `anl69_roa_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 44 | 0.786170 | `False` |
| `anl69_roa_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 2 | 0.710641 | `False` |
| `anl69_roa_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 2 | 0.710641 | `False` |
| `anl69_roe_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 11 | 0.749305 | `False` |
| `anl69_roe_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9935 | 7 | 0.734747 | `False` |
| `anl69_roe_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6029 | 1 | 0.500782 | `False` |
| `anl69_roe_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 3 | 0.718665 | `False` |
| `anl69_roe_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 17 | 0.760614 | `False` |
| `anl69_roe_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 16 | 0.759020 | `False` |
| `anl69_roe_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 7 | 0.737997 | `False` |
| `anl69_roe_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 5 | 0.729973 | `False` |
| `anl69_roes_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 1 | 0.699332 | `False` |
| `anl69_roes_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9935 | 6 | 0.731023 | `False` |
| `anl69_roes_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6029 | 0 | 0.481450 | `False` |
| `anl69_roes_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 13 | 0.753605 | `False` |
| `anl69_roes_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 24 | 0.769776 | `False` |
| `anl69_roes_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 12 | 0.751538 | `False` |
| `anl69_roes_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 2 | 0.710641 | `False` |
| `anl69_roes_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 3 | 0.718665 | `False` |
| `anl69_sales_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 9 | 0.744220 | `False` |
| `anl69_sales_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 9 | 0.740120 | `False` |
| `anl69_sales_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 2 | 0.511241 | `False` |
| `anl69_sales_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 13 | 0.753605 | `False` |
| `anl69_sales_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 13 | 0.753605 | `False` |
| `anl69_sales_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 82 | 0.803244 | `False` |
| `anl69_sales_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 7 | 0.737997 | `False` |
| `anl69_sales_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 3 | 0.718665 | `False` |
| `anl69_saless_best_crncy_iso` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 1 | 0.699332 | `False` |
| `anl69_saless_best_cur_fiscal_qtr_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.9918 | 27 | 0.768837 | `False` |
| `anl69_saless_best_cur_fiscal_semi_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 0.6012 | 4 | 0.525488 | `False` |
| `anl69_saless_best_cur_fiscal_year_period` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 14 | 0.755529 | `False` |
| `anl69_saless_best_eeps_cur_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 36 | 0.780711 | `False` |
| `anl69_saless_best_eeps_nxt_yr` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 21 | 0.766211 | `False` |
| `anl69_saless_expected_report_dt` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 6 | 0.734273 | `False` |
| `anl69_saless_expected_report_time` | `Fundamental Analyst Estimates` | `analyst` | `GLB` | `TOP3000` | 1.0000 | 18 | 0.762122 | `False` |
| `None` | `` | `other` | `` | `` | 0.0000 | 0 | 0.100000 | `False` |
| `beta` | `runtime` | `other` | `` | `` | 1.0000 | 0 | 0.600000 | `True` |
| `liquidity` | `runtime` | `other` | `` | `` | 1.0000 | 0 | 0.600000 | `True` |
| `size` | `runtime` | `other` | `` | `` | 1.0000 | 0 | 0.600000 | `True` |
| `volatility` | `runtime` | `other` | `` | `` | 1.0000 | 0 | 0.600000 | `True` |
| `close` | `runtime` | `price` | `` | `` | 1.0000 | 0 | 0.700000 | `True` |
| `high` | `runtime` | `price` | `` | `` | 1.0000 | 0 | 0.700000 | `True` |
| `low` | `runtime` | `price` | `` | `` | 1.0000 | 0 | 0.700000 | `True` |
| `open` | `runtime` | `price` | `` | `` | 1.0000 | 0 | 0.700000 | `True` |
| `returns` | `runtime` | `price` | `` | `` | 0.9844 | 0 | 0.692187 | `True` |
| `volume` | `runtime` | `volume` | `` | `` | 1.0000 | 0 | 0.670000 | `True` |

## Currently Eligible Group/Vector Fields

These are treated as group/vector fields, not normal numeric matrix inputs. Use mainly with group operators if appropriate.

### `Analyst Trade Ideas` (57)

- `anl45_ad_rel_ret_per`
- `anl45_ad_ret_per`
- `anl45_ang_inv`
- `anl45_avg_dur`
- `anl45_avg_initial_prc`
- `anl45_avg_initial_prc_wfee`
- `anl45_beta`
- `anl45_bias_weighted_ret`
- `anl45_bm_exchange_rate`
- `anl45_bm_fx_ret`
- `anl45_bm_ret`
- `anl45_bm_ret_wo_fx`
- `anl45_closed_prc`
- `anl45_current_inv`
- `anl45_days_since_inception`
- `anl45_idea_count`
- `anl45_index_period_end_prc`
- `anl45_index_period_start_prc`
- `anl45_index_ret_per`
- `anl45_initial_inv`
- `anl45_inv_exchange_rate`
- `anl45_jensensalpha`
- `anl45_latest_prc`
- `anl45_net_market_exposure`
- `anl45_new_value`
- `anl45_old_value`
- `anl45_period_end_prc`
- `anl45_period_start_prc`
- `anl45_prc`
- `anl45_prc_change_per_today`
- `anl45_prc_change_today`
- `anl45_prev_close_prc`
- `anl45_real_ret`
- `anl45_real_ret_today`
- `anl45_real_value`
- `anl45_rel_index_ret`
- `anl45_rel_index_ret_per`
- `anl45_rel_ret_per_today`
- `anl45_rel_ret_today`
- `anl45_ret_per_today`
- `anl45_ret_today`
- `anl45_risk_free_rate`
- `anl45_stock_ret_per`
- `anl45_stock_ret_per_relative`
- `anl45_target_prc`
- `anl45_time`
- `anl45_tot_ret`
- `anl45_tot_ret_per`
- `anl45_tot_ret_wo_fx`
- `anl45_transaction_charge`
- `anl45_treynor_ratio`
- `anl45_unreal_ret`
- `anl45_unreal_ret_today`
- `anl45_unreal_value`
- `benchmark_currency_code`
- `investment_currency_code`
- `security_trading_currency_3`

### `Analyst estimates & financial ratios` (267)

- `annual_period_count`
- `annual_period_count_dup`
- `cash_flow_per_share_quarterly_prev_year`
- `debt_to_equity_ratio_quarterly_prior_year`
- `depreciation_expense_quarterly`
- `eps_excl_extra_items_quarterly`
- `free_cash_flow_per_share_quarterly_prev_year`
- `free_cash_flow_quarterly`
- `income_taxes_paid_quarterly`
- `lowest_pe_ratio_quarterly`
- `price_to_cash_flow_per_share_quarterly`
- `quarter_total_assets`
- `quarterly_accumulated_depreciation`
- `quarterly_accumulated_depreciation_2`
- `quarterly_accumulated_depreciation_db`
- `quarterly_asset_turnover`
- `quarterly_asset_turnover_2`
- `quarterly_asset_turnover_db`
- `quarterly_basic_eps_excl_extra_items`
- `quarterly_basic_eps_excl_extraordinary`
- `quarterly_basic_eps_excl_extraordinary_dup`
- `quarterly_book_value_per_share`
- `quarterly_book_value_per_share_2`
- `quarterly_book_value_per_share_3`
- `quarterly_capital_spending_per_share`
- `quarterly_capital_spending_per_share_2`
- `quarterly_capital_spending_per_share_3`
- `quarterly_cash`
- `quarterly_cash_and_equivalents`
- `quarterly_cash_balance`
- `quarterly_cash_flow_per_share`
- `quarterly_cash_flow_per_share_2`
- `quarterly_cash_flow_per_share_3`
- `quarterly_cash_flow_per_share_alt`
- `quarterly_cash_flow_per_share_alt_2`
- `quarterly_cash_flow_per_share_non_annualized`
- `quarterly_cash_flow_per_share_non_annualized_alt`
- `quarterly_cash_flow_per_share_nonannualized`
- `quarterly_cash_flow_per_share_nonannualized_prior`
- `quarterly_cash_per_share`
- `quarterly_cash_per_share_2`
- `quarterly_cash_per_share_3`
- `quarterly_common_equity`
- `quarterly_common_equity_2`
- `quarterly_common_equity_3`
- `quarterly_common_equity_prior`
- `quarterly_common_equity_prior_year`
- `quarterly_common_equity_prior_year_2`
- `quarterly_cost_of_goods_sold`
- `quarterly_cost_of_goods_sold_2`
- `quarterly_cost_of_goods_sold_3`
- `quarterly_current_assets`
- `quarterly_current_assets_2`
- `quarterly_current_assets_3`
- `quarterly_current_liabilities`
- `quarterly_current_liabilities_2`
- `quarterly_current_liabilities_3`
- `quarterly_current_ratio`
- `quarterly_current_ratio_2`
- `quarterly_current_ratio_3`
- `quarterly_current_ratio_alt`
- `quarterly_current_ratio_prior`
- `quarterly_current_ratio_prior_year`
- `quarterly_debt_service_to_eps`
- `quarterly_debt_service_to_eps_2`
- `quarterly_debt_service_to_eps_3`
- `quarterly_depreciation_expense`
- `quarterly_depreciation_expense_2`
- `quarterly_dividend_payout_ratio`
- `quarterly_dividend_per_share`
- `quarterly_dividend_per_share_2`
- `quarterly_dividend_per_share_3`
- `quarterly_earnings_before_tax`
- `quarterly_earnings_before_tax_2`
- `quarterly_earnings_before_tax_3`
- `quarterly_ebit`
- `quarterly_ebit_2`
- `quarterly_ebit_3`
- `quarterly_ebitda`
- `quarterly_ebitda_2`
- `quarterly_ebitda_db`
- `quarterly_effective_tax_rate`
- `quarterly_effective_tax_rate_dup`
- `quarterly_eps_excl_extraordinary`
- `quarterly_eps_excluding_extraordinary`
- `quarterly_eps_incl_extraordinary`
- `quarterly_eps_including_extra_items`
- `quarterly_eps_including_extraordinary`
- `quarterly_free_cash_flow`
- `quarterly_free_cash_flow_2`
- `quarterly_free_cash_flow_per_share`
- `quarterly_free_cash_flow_per_share_2`
- `quarterly_free_cash_flow_per_share_3`
- `quarterly_free_cash_flow_per_share_alt`
- `quarterly_free_cash_flow_per_share_alt_2`
- `quarterly_free_cash_flow_per_share_non_annualized`
- `quarterly_free_cash_flow_per_share_non_annualized_2`
- `quarterly_free_cash_flow_per_share_non_annualized_alt`
- `quarterly_free_cash_flow_per_share_prior`
- `quarterly_gross_margin_percent`
- `quarterly_gross_margin_percent_2`
- `quarterly_gross_margin_percent_3`
- `quarterly_interest_coverage`
- `quarterly_interest_coverage_db`
- `quarterly_interest_coverage_ratio`
- `quarterly_interest_expense`
- `quarterly_interest_expense_2`
- `quarterly_interest_expense_3`
- `quarterly_inventory`
- `quarterly_inventory_2`
- `quarterly_inventory_turnover`
- `quarterly_inventory_turnover_2`
- `quarterly_inventory_turnover_ratio`
- `quarterly_inventory_value`
- `quarterly_long_term_debt`
- `quarterly_long_term_debt_2`
- `quarterly_long_term_debt_3`
- `quarterly_long_term_debt_per_share`
- `quarterly_long_term_debt_per_share_2`
- `quarterly_long_term_debt_per_share_3`
- `quarterly_long_term_debt_to_assets`
- `quarterly_long_term_debt_to_assets_2`
- `quarterly_long_term_debt_to_assets_ratio`
- `quarterly_long_term_debt_to_capital`
- `quarterly_long_term_debt_to_equity`
- `quarterly_long_term_debt_to_equity_2`
- `quarterly_long_term_debt_to_equity_prior`
- `quarterly_long_term_debt_to_equity_prior_year`
- `quarterly_long_term_debt_to_equity_ratio`
- `quarterly_long_term_debt_to_equity_ratio_prior_year`
- `quarterly_long_term_debt_to_total_capital`
- `quarterly_long_term_debt_to_total_capital_2`
- `quarterly_net_income`
- `quarterly_net_income_2`
- `quarterly_net_income_available_to_common`
- `quarterly_net_income_available_to_common_2`
- `quarterly_net_income_available_to_common_3`
- `quarterly_net_income_dup`
- `quarterly_net_income_per_employee`
- `quarterly_net_income_per_employee_2`
- `quarterly_net_income_per_employee_3`
- `quarterly_net_loans`
- `quarterly_net_loans_2`
- `quarterly_net_loans_3`
- `quarterly_net_loans_change_percent`
- `quarterly_net_loans_change_prior_year`
- `quarterly_net_loans_change_yoy`
- `quarterly_net_profit_margin_percent`
- `quarterly_net_profit_margin_percent_2`
- `quarterly_net_profit_margin_percent_3`
- `quarterly_non_annualized_cash_flow_per_share`
- `quarterly_non_annualized_cash_flow_per_share_alt`
- `quarterly_non_annualized_fcf_per_share`
- `quarterly_non_annualized_fcf_per_share_alt`
- `quarterly_non_annualized_revenue_per_share`
- `quarterly_operating_margin_percent`
- `quarterly_operating_margin_percent_2`
- `quarterly_operating_margin_percent_3`
- `quarterly_payout_ratio`
- `quarterly_payout_ratio_2`
- `quarterly_pe_ratio_high`
- `quarterly_pe_ratio_high_2`
- `quarterly_pe_ratio_low`
- `quarterly_pe_ratio_maximum`
- `quarterly_pe_ratio_minimum`
- `quarterly_pretax_margin_percent`
- `quarterly_pretax_margin_percent_2`
- `quarterly_pretax_margin_percent_3`
- `quarterly_price_to_cash_flow_per_share`
- `quarterly_price_to_cash_flow_per_share_2`
- `quarterly_price_to_sales_ratio`
- `quarterly_price_to_sales_ratio_2`
- `quarterly_price_to_sales_ratio_3`
- `quarterly_quick_ratio`
- `quarterly_quick_ratio_2`
- `quarterly_quick_ratio_3`
- `quarterly_quick_ratio_alt`
- `quarterly_quick_ratio_prior`
- `quarterly_quick_ratio_prior_year`
- `quarterly_receivables`
- `quarterly_receivables_2`
- `quarterly_receivables_db`
- `quarterly_receivables_turnover`
- `quarterly_receivables_turnover_2`
- `quarterly_reinvestment_rate`
- `quarterly_reinvestment_rate_2`
- `quarterly_reinvestment_rate_3`
- `quarterly_research_and_development_expense_2`
- `quarterly_research_and_development_expense_3`
- `quarterly_research_development_expense`
- `quarterly_return_on_assets_percent`
- `quarterly_return_on_assets_percent_2`
- `quarterly_return_on_average_assets`
- `quarterly_return_on_average_equity`
- `quarterly_return_on_equity_percent`
- `quarterly_return_on_equity_percent_2`
- `quarterly_return_on_investment`
- `quarterly_return_on_investment_percent`
- `quarterly_return_on_investment_percent_2`
- `quarterly_revenue`
- `quarterly_revenue_per_employee`
- `quarterly_revenue_per_employee_2`
- `quarterly_revenue_per_employee_3`
- `quarterly_revenue_per_share`
- `quarterly_revenue_per_share_2`
- `quarterly_revenue_per_share_3`
- `quarterly_revenue_per_share_non_annualized`
- `quarterly_revenue_per_share_nonannualized`
- `quarterly_sga_to_revenue_ratio`
- `quarterly_sga_to_sales_ratio`
- `quarterly_sga_to_sales_ratio_2`
- `quarterly_shareholder_equity`
- `quarterly_tangible_book_value`
- `quarterly_tangible_book_value_2`
- `quarterly_tangible_book_value_3`
- `quarterly_tangible_book_value_per_share`
- `quarterly_tangible_book_value_per_share_2`
- `quarterly_tangible_book_value_per_share_3`
- `quarterly_tax_rate_percent`
- `quarterly_taxes_paid`
- `quarterly_taxes_paid_2`
- `quarterly_total_assets`
- `quarterly_total_assets_change_percent`
- `quarterly_total_assets_change_prior_year`
- `quarterly_total_assets_change_yoy`
- `quarterly_total_assets_db`
- `quarterly_total_debt`
- `quarterly_total_debt_2`
- `quarterly_total_debt_3`
- `quarterly_total_debt_to_assets`
- `quarterly_total_debt_to_assets_2`
- `quarterly_total_debt_to_assets_ratio`
- `quarterly_total_debt_to_capital`
- `quarterly_total_debt_to_capital_2`
- `quarterly_total_debt_to_equity`
- `quarterly_total_debt_to_equity_2`
- `quarterly_total_debt_to_equity_prior_year`
- `quarterly_total_debt_to_equity_ratio`
- `quarterly_total_debt_to_equity_ratio_prior_year`
- `quarterly_total_debt_to_total_capital_ratio`
- `quarterly_total_liabilities`
- `quarterly_total_liabilities_2`
- `quarterly_total_liabilities_3`
- `quarterly_total_revenue`
- `quarterly_total_revenue_2`
- `quarterly_total_shareholder_equity`
- `quarterly_total_shareholder_equity_2`
- `quarterly_working_capital_per_share_to_price`
- `quarterly_working_capital_per_share_to_price_2`
- `quarterly_working_capital_per_share_to_price_db`
- `receivables_turnover_quarterly`
- `reporting_to_pricing_currency_exchange_rate`
- `reporting_to_pricing_currency_exrate`
- `ttm_debt_service_to_eps`
- `ttm_debt_service_to_eps_2`
- `ttm_debt_service_to_eps_3`
- `ttm_ebit`
- `ttm_ebit_2`
- `ttm_ebit_3`
- `ttm_price_to_free_cash_flow_per_share`
- `ttm_price_to_free_cash_flow_per_share_2`
- `ttm_price_to_free_cash_flow_per_share_3`
- `ttm_return_on_average_equity`
- `ttm_return_on_equity_percent`
- `ttm_return_on_equity_percent_2`
- `usd_to_reporting_currency_exchange_rate`
- `usd_to_reporting_currency_exrate`

### `Fundamental Analyst Estimates` (204)

- `anl69_best_cur_ev_to_ebitda`
- `anl69_best_ebit`
- `anl69_best_ebit_4wk_chg`
- `anl69_best_ebit_4wk_dn`
- `anl69_best_ebit_4wk_up`
- `anl69_best_ebit_chg_pct`
- `anl69_best_ebit_hi`
- `anl69_best_ebit_lo`
- `anl69_best_ebit_median`
- `anl69_best_ebit_numest`
- `anl69_best_ebit_stddev`
- `anl69_best_ebit_to_sales`
- `anl69_best_eps_4wk_dn`
- `anl69_best_eps_4wk_up`
- `anl69_best_eps_chg_pct`
- `anl69_best_eps_gaap_4wk_dn`
- `anl69_best_eps_gaap_4wk_up`
- `anl69_best_eps_gaap_median`
- `anl69_best_eps_median`
- `anl69_best_ev_to_best_ebit`
- `anl69_best_ndebt_4wk_chg`
- `anl69_best_ndebt_4wk_dn`
- `anl69_best_ndebt_4wk_up`
- `anl69_best_ndebt_chg_pct`
- `anl69_best_ndebt_hi`
- `anl69_best_ndebt_lo`
- `anl69_best_ndebt_median`
- `anl69_best_ndebt_numest`
- `anl69_best_ndebt_stddev`
- `anl69_best_net_4wk_chg`
- `anl69_best_net_4wk_dn`
- `anl69_best_net_4wk_up`
- `anl69_best_net_chg_pct`
- `anl69_best_net_debt`
- `anl69_best_net_gaap`
- `anl69_best_net_gaap_4wk_chg`
- `anl69_best_net_gaap_4wk_dn`
- `anl69_best_net_gaap_4wk_up`
- `anl69_best_net_gaap_hi`
- `anl69_best_net_gaap_lo`
- `anl69_best_net_gaap_median`
- `anl69_best_net_gaap_numest`
- `anl69_best_net_gaap_stddev`
- `anl69_best_net_hi`
- `anl69_best_net_income`
- `anl69_best_net_lo`
- `anl69_best_net_median`
- `anl69_best_net_numest`
- `anl69_best_net_stddev`
- `anl69_best_pe_ratio`
- `anl69_best_px_bps_ratio`
- `anl69_best_px_cps_ratio`
- `anl69_best_roa`
- `anl69_best_roa_4wk_chg`
- `anl69_best_roa_4wk_dn`
- `anl69_best_roa_4wk_up`
- `anl69_best_roa_chg_pct`
- `anl69_best_roa_hi`
- `anl69_best_roa_lo`
- `anl69_best_roa_median`
- `anl69_best_roa_numest`
- `anl69_best_roa_stddev`
- `anl69_cps_best_cps`
- `anl69_cps_best_cps_4wk_chg`
- `anl69_cps_best_cps_4wk_dn`
- `anl69_cps_best_cps_4wk_up`
- `anl69_cps_best_cps_chg_pct`
- `anl69_cps_best_cps_hi`
- `anl69_cps_best_cps_lo`
- `anl69_cps_best_cps_median`
- `anl69_cps_best_cps_numest`
- `anl69_cps_best_cps_stddev`
- `anl69_cps_best_fiscal_period_dt`
- `anl69_cps_best_fperiod_override`
- `anl69_cps_latest_ann_dt_qtrly`
- `anl69_cps_most_recent_period_end_dt`
- `anl69_cpss_best_cps`
- `anl69_cpss_best_cps_4wk_chg`
- `anl69_cpss_best_cps_4wk_dn`
- `anl69_cpss_best_cps_4wk_up`
- `anl69_cpss_best_cps_chg_pct`
- `anl69_cpss_best_cps_hi`
- `anl69_cpss_best_cps_lo`
- `anl69_cpss_best_cps_median`
- `anl69_cpss_best_cps_numest`
- `anl69_cpss_best_cps_stddev`
- `anl69_cpss_best_fiscal_period_dt`
- `anl69_cpss_best_fperiod_override`
- `anl69_cpss_latest_ann_dt_qtrly`
- `anl69_cpss_most_recent_period_end_dt`
- `anl69_dps_best_dps`
- `anl69_dps_best_dps_4wk_chg`
- `anl69_dps_best_dps_4wk_dn`
- `anl69_dps_best_dps_4wk_up`
- `anl69_dps_best_dps_hi`
- `anl69_dps_best_dps_lo`
- `anl69_dps_best_dps_median`
- `anl69_dps_best_dps_numest`
- `anl69_dps_best_dps_stddev`
- `anl69_dps_best_fiscal_period_dt`
- `anl69_dps_best_fperiod_override`
- `anl69_dps_latest_ann_dt_qtrly`
- `anl69_dps_most_recent_period_end_dt`
- `anl69_dpss_best_dps`
- `anl69_dpss_best_dps_4wk_chg`
- `anl69_dpss_best_dps_4wk_dn`
- `anl69_dpss_best_dps_4wk_up`
- `anl69_dpss_best_dps_hi`
- `anl69_dpss_best_dps_lo`
- `anl69_dpss_best_dps_median`
- `anl69_dpss_best_dps_numest`
- `anl69_dpss_best_dps_stddev`
- `anl69_dpss_best_fiscal_period_dt`
- `anl69_dpss_best_fperiod_override`
- `anl69_dpss_latest_ann_dt_qtrly`
- `anl69_dpss_most_recent_period_end_dt`
- `anl69_ebit_best_fiscal_period_dt`
- `anl69_ebit_best_fperiod_override`
- `anl69_ebit_latest_ann_dt_qtrly`
- `anl69_ebit_most_recent_period_end_dt`
- `anl69_eps_best_eps`
- `anl69_eps_best_eps_4wk_chg`
- `anl69_eps_best_eps_gaap`
- `anl69_eps_best_eps_gaap_4wk_chg`
- `anl69_eps_best_eps_gaap_hi`
- `anl69_eps_best_eps_gaap_lo`
- `anl69_eps_best_eps_gaap_numest`
- `anl69_eps_best_eps_gaap_stddev`
- `anl69_eps_best_eps_hi`
- `anl69_eps_best_eps_lo`
- `anl69_eps_best_eps_numest`
- `anl69_eps_best_eps_stddev`
- `anl69_eps_best_fiscal_period_dt`
- `anl69_eps_best_fperiod_override`
- `anl69_eps_latest_ann_dt_qtrly`
- `anl69_eps_most_recent_period_end_dt`
- `anl69_epss_best_eps`
- `anl69_epss_best_eps_4wk_chg`
- `anl69_epss_best_eps_gaap`
- `anl69_epss_best_eps_gaap_4wk_chg`
- `anl69_epss_best_eps_gaap_hi`
- `anl69_epss_best_eps_gaap_lo`
- `anl69_epss_best_eps_gaap_numest`
- `anl69_epss_best_eps_gaap_stddev`
- `anl69_epss_best_eps_hi`
- `anl69_epss_best_eps_lo`
- `anl69_epss_best_eps_numest`
- `anl69_epss_best_eps_stddev`
- `anl69_epss_best_fperiod_override`
- `anl69_ndebt_best_fiscal_period_dt`
- `anl69_ndebt_best_fperiod_override`
- `anl69_ndebt_latest_ann_dt_qtrly`
- `anl69_ndebt_most_recent_period_end_dt`
- `anl69_net_best_fiscal_period_dt`
- `anl69_net_best_fperiod_override`
- `anl69_net_latest_ann_dt_qtrly`
- `anl69_net_most_recent_period_end_dt`
- `anl69_pe_best_fperiod_override`
- `anl69_roa_best_fiscal_period_dt`
- `anl69_roa_best_fperiod_override`
- `anl69_roa_latest_ann_dt_qtrly`
- `anl69_roa_most_recent_period_end_dt`
- `anl69_roe_best_fiscal_period_dt`
- `anl69_roe_best_fperiod_override`
- `anl69_roe_best_roe`
- `anl69_roe_best_roe_4wk_chg`
- `anl69_roe_best_roe_4wk_dn`
- `anl69_roe_best_roe_4wk_up`
- `anl69_roe_best_roe_chg_pct`
- `anl69_roe_best_roe_hi`
- `anl69_roe_best_roe_lo`
- `anl69_roe_best_roe_median`
- `anl69_roe_best_roe_numest`
- `anl69_roe_best_roe_stddev`
- `anl69_roe_latest_ann_dt_qtrly`
- `anl69_roe_most_recent_period_end_dt`
- `anl69_roes_best_fiscal_period_dt`
- `anl69_roes_best_fperiod_override`
- `anl69_roes_best_roe`
- `anl69_roes_best_roe_4wk_chg`
- `anl69_roes_best_roe_4wk_dn`
- `anl69_roes_best_roe_4wk_up`
- `anl69_roes_best_roe_chg_pct`
- `anl69_roes_best_roe_hi`
- `anl69_roes_best_roe_lo`
- `anl69_roes_best_roe_median`
- `anl69_roes_best_roe_numest`
- `anl69_roes_best_roe_stddev`
- `anl69_roes_latest_ann_dt_qtrly`
- `anl69_roes_most_recent_period_end_dt`
- `anl69_sales_best_fperiod_override`
- `anl69_sales_best_sales`
- `anl69_sales_best_sales_4wk_chg`
- `anl69_sales_best_sales_hi`
- `anl69_sales_best_sales_lo`
- `anl69_sales_best_sales_numest`
- `anl69_sales_best_sales_stddev`
- `anl69_saless_best_fperiod_override`
- `anl69_saless_best_sales`
- `anl69_saless_best_sales_4wk_chg`
- `anl69_saless_best_sales_hi`
- `anl69_saless_best_sales_lo`
- `anl69_saless_best_sales_numest`
- `anl69_saless_best_sales_stddev`

### `Integrated Broker Estimates` (138)

- `anl44_2_bps_coveredby`
- `anl44_2_bps_lastactccy`
- `anl44_2_bps_lastactvalue`
- `anl44_2_bps_prevalue`
- `anl44_2_bps_value`
- `anl44_2_capex_coveredby`
- `anl44_2_capex_lastactccy`
- `anl44_2_capex_lastactvalue`
- `anl44_2_capex_prevalue`
- `anl44_2_capex_value`
- `anl44_2_cfps_coveredby`
- `anl44_2_cfps_lastactccy`
- `anl44_2_cfps_lastactvalue`
- `anl44_2_cfps_prevalue`
- `anl44_2_cfps_value`
- `anl44_2_csh_coveredby`
- `anl44_2_csh_lastactccy`
- `anl44_2_csh_lastactvalue`
- `anl44_2_csh_prevalue`
- `anl44_2_csh_value`
- `anl44_2_dps_coveredby`
- `anl44_2_dps_lastactccy`
- `anl44_2_dps_lastactvalue`
- `anl44_2_dps_prevalue`
- `anl44_2_dps_value`
- `anl44_2_ebit_coveredby`
- `anl44_2_ebit_lastactccy`
- `anl44_2_ebit_lastactvalue`
- `anl44_2_ebit_prevalue`
- `anl44_2_ebit_value`
- `anl44_2_ebitda_coveredby`
- `anl44_2_ebitda_lastactccy`
- `anl44_2_ebitda_lastactvalue`
- `anl44_2_ebitda_prevalue`
- `anl44_2_ebitda_value`
- `anl44_2_ebitdaps_coveredby`
- `anl44_2_ebitdaps_lastactccy`
- `anl44_2_ebitdaps_lastactvalue`
- `anl44_2_ebitdaps_prevalue`
- `anl44_2_ebitdaps_value`
- `anl44_2_eps_coveredby`
- `anl44_2_eps_lastactccy`
- `anl44_2_eps_lastactvalue`
- `anl44_2_eps_prevalue`
- `anl44_2_eps_value`
- `anl44_2_epsr_coveredby`
- `anl44_2_epsr_lastactccy`
- `anl44_2_epsr_lastactvalue`
- `anl44_2_epsr_prevalue`
- `anl44_2_epsr_value`
- `anl44_2_fcfps_coveredby`
- `anl44_2_fcfps_lastactccy`
- `anl44_2_fcfps_lastactvalue`
- `anl44_2_fcfps_prevalue`
- `anl44_2_fcfps_value`
- `anl44_2_grossmargin_coveredby`
- `anl44_2_grossmargin_lastactccy`
- `anl44_2_grossmargin_lastactvalue`
- `anl44_2_grossmargin_prevalue`
- `anl44_2_grossmargin_value`
- `anl44_2_nav_coveredby`
- `anl44_2_nav_lastactccy`
- `anl44_2_nav_lastactvalue`
- `anl44_2_nav_prevalue`
- `anl44_2_nav_value`
- `anl44_2_netdebt_coveredby`
- `anl44_2_netdebt_lastactccy`
- `anl44_2_netdebt_lastactvalue`
- `anl44_2_netdebt_prevalue`
- `anl44_2_netdebt_value`
- `anl44_2_netprofit_coveredby`
- `anl44_2_netprofit_lastactccy`
- `anl44_2_netprofit_lastactvalue`
- `anl44_2_netprofit_prevalue`
- `anl44_2_netprofit_rep_coveredby`
- `anl44_2_netprofit_rep_lastactccy`
- `anl44_2_netprofit_rep_lastactvalue`
- `anl44_2_netprofit_rep_prevalue`
- `anl44_2_netprofit_rep_value`
- `anl44_2_netprofit_value`
- `anl44_2_operatingprofit_coveredby`
- `anl44_2_operatingprofit_lastactccy`
- `anl44_2_operatingprofit_lastactvalue`
- `anl44_2_operatingprofit_prevalue`
- `anl44_2_operatingprofit_value`
- `anl44_2_pretaxprofit_coveredby`
- `anl44_2_pretaxprofit_lastactccy`
- `anl44_2_pretaxprofit_lastactvalue`
- `anl44_2_pretaxprofit_prevalue`
- `anl44_2_pretaxprofit_rep_coveredby`
- `anl44_2_pretaxprofit_rep_lastactccy`
- `anl44_2_pretaxprofit_rep_lastactvalue`
- `anl44_2_pretaxprofit_rep_prevalue`
- `anl44_2_pretaxprofit_rep_value`
- `anl44_2_pretaxprofit_value`
- `anl44_2_roa_coveredby`
- `anl44_2_roa_lastactccy`
- `anl44_2_roa_lastactvalue`
- `anl44_2_roa_prevalue`
- `anl44_2_roa_value`
- `anl44_2_roe_coveredby`
- `anl44_2_roe_lastactccy`
- `anl44_2_roe_lastactvalue`
- `anl44_2_roe_prevalue`
- `anl44_2_roe_value`
- `anl44_2_sales_coveredby`
- `anl44_2_sales_lastactccy`
- `anl44_2_sales_lastactvalue`
- `anl44_2_sales_prevalue`
- `anl44_2_sales_value`
- `anl44_2_tbvps_coveredby`
- `anl44_2_tbvps_lastactccy`
- `anl44_2_tbvps_lastactvalue`
- `anl44_2_tbvps_prevalue`
- `anl44_2_tbvps_value`
- `capex_currency_code`
- `cash_eps_currency_code_2`
- `cfps_currency_code`
- `ebit_currency_code`
- `ebitda_currency_code`
- `ebitdaps_currency_code`
- `eps_currency_code_2`
- `epsr_currency_code`
- `forecast_currency_book_value_per_share`
- `forecast_currency_code_2`
- `forecast_currency_dps`
- `forecast_currency_fcfps`
- `forecast_currency_netprofit`
- `forecast_currency_roa`
- `forecast_currency_roe`
- `forecast_currency_sales`
- `forecast_currency_tangible_book_value_per_share`
- `grossmargin_currency_code`
- `nav_currency_code_3`
- `netdebt_currency_code`
- `operatingprofit_currency_code`
- `pretaxprofit_currency_code`
- `pretaxprofitrep_currency_code`

### `runtime` (4)

- `country`
- `industry`
- `sector`
- `subindustry`

## Recent Result Context

After hotfix round >= 12754: total results=135, completed=42, timeout=93, avg_sharpe=0.29928571428571427, avg_fitness=0.16333333333333333.
Latest closed-loop round in DB: `12769`.

## Historical Run Caveat

The current run has 82308 historical alpha rows and 10647 distinct fields because this service run spans many older configs. Do not use that full historical field set as the current search-space unless explicitly analyzing legacy behavior.

