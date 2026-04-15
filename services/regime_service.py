from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import LearnedRegimeConfig, RegimeDetectionConfig
from services.models import RegimeSnapshot

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime fallback when deps are unavailable
    MiniBatchKMeans = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

try:
    from tsfresh.feature_extraction import MinimalFCParameters, extract_features

    TSFRESH_AVAILABLE = True
    TSFRESH_PARAMETERS = MinimalFCParameters()
except ImportError:  # pragma: no cover - runtime fallback when deps are unavailable
    extract_features = None
    TSFRESH_AVAILABLE = False
    TSFRESH_PARAMETERS = None


@dataclass(frozen=True, slots=True)
class _HeuristicRegimeState:
    market_regime_key: str
    regime_label: str
    confidence: float
    features: dict[str, float | int | str]


class RegimeService:
    """Infer a market-state regime while preserving the legacy/fallback behavior."""

    def resolve(
        self,
        *,
        matrices,
        config: RegimeDetectionConfig,
        region: str,
        legacy_regime_key: str,
        global_regime_key: str,
        learned_config: LearnedRegimeConfig | None = None,
    ) -> RegimeSnapshot:
        if not config.enabled:
            return RegimeSnapshot(
                region=region,
                legacy_regime_key=legacy_regime_key,
                global_regime_key=global_regime_key,
                market_regime_key="",
                effective_regime_key=legacy_regime_key,
                regime_label="unknown",
                confidence=0.0,
                features={},
            )

        returns_frame = matrices.numeric_fields.get("returns")
        if returns_frame is None or returns_frame.empty:
            return self._fallback(region=region, legacy_regime_key=legacy_regime_key, global_regime_key=global_regime_key)

        market_proxy = returns_frame.mean(axis=1).dropna()
        cross_sectional_dispersion = returns_frame.std(axis=1, ddof=0).dropna()
        if market_proxy.empty or cross_sectional_dispersion.empty:
            return self._fallback(region=region, legacy_regime_key=legacy_regime_key, global_regime_key=global_regime_key)

        common_index = market_proxy.index.intersection(cross_sectional_dispersion.index)
        market_proxy = market_proxy.loc[common_index].dropna()
        cross_sectional_dispersion = cross_sectional_dispersion.loc[common_index].dropna()
        common_index = market_proxy.index.intersection(cross_sectional_dispersion.index)
        market_proxy = market_proxy.loc[common_index]
        cross_sectional_dispersion = cross_sectional_dispersion.loc[common_index]
        returns_frame = returns_frame.loc[common_index]
        if len(common_index) < int(config.min_points):
            return self._fallback(region=region, legacy_regime_key=legacy_regime_key, global_regime_key=global_regime_key)

        heuristic = self._heuristic_regime(
            market_proxy=market_proxy,
            cross_sectional_dispersion=cross_sectional_dispersion,
            config=config,
        )
        learned = self._learned_regime(
            market_proxy=market_proxy,
            cross_sectional_dispersion=cross_sectional_dispersion,
            returns_frame=returns_frame,
            config=learned_config or LearnedRegimeConfig(),
        )
        market_regime_key = heuristic.market_regime_key
        regime_label = heuristic.regime_label
        confidence = heuristic.confidence
        features = dict(heuristic.features)
        features["heuristic_market_regime_key"] = heuristic.market_regime_key
        features["heuristic_regime_label"] = heuristic.regime_label
        features["heuristic_confidence"] = heuristic.confidence
        if learned is not None:
            features.update(learned)
            learned_confidence = float(learned.get("learned_confidence", 0.0) or 0.0)
            cluster_id = learned.get("learned_cluster_id")
            if cluster_id not in {None, ""} and learned_confidence >= float((learned_config or LearnedRegimeConfig()).confidence_floor):
                market_regime_key = f"learned_cluster:{cluster_id}"
                regime_label = f"learned_cluster_{cluster_id}"
                confidence = learned_confidence
        effective_regime_key = legacy_regime_key
        if confidence >= float(config.min_confidence):
            payload = json.dumps(
                {
                    "legacy_regime_key": legacy_regime_key,
                    "market_regime_key": market_regime_key,
                },
                sort_keys=True,
            )
            effective_regime_key = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return RegimeSnapshot(
            region=region,
            legacy_regime_key=legacy_regime_key,
            global_regime_key=global_regime_key,
            market_regime_key=market_regime_key,
            effective_regime_key=effective_regime_key,
            regime_label=regime_label,
            confidence=confidence,
            features=features,
        )

    def _heuristic_regime(
        self,
        *,
        market_proxy: pd.Series,
        cross_sectional_dispersion: pd.Series,
        config: RegimeDetectionConfig,
    ) -> _HeuristicRegimeState:
        short_window = int(max(2, config.short_window))
        long_window = int(max(short_window, config.long_window))
        short_market = market_proxy.tail(short_window)
        long_market = market_proxy.tail(long_window)
        short_dispersion = cross_sectional_dispersion.tail(short_window)
        long_dispersion = cross_sectional_dispersion.tail(long_window)

        short_vol = float(short_market.std(ddof=0) or 0.0)
        long_vol = float(long_market.std(ddof=0) or 0.0)
        vol_ratio = short_vol / max(long_vol, 1e-9)

        short_mean = float(short_market.mean() or 0.0)
        trend_scale = max(long_vol / math.sqrt(max(1.0, float(short_window))), 1e-9)
        trend_z = short_mean / trend_scale

        short_dispersion_mean = float(short_dispersion.mean() or 0.0)
        long_dispersion_mean = float(long_dispersion.mean() or 0.0)
        dispersion_ratio = short_dispersion_mean / max(long_dispersion_mean, 1e-9)

        volatility_regime = "normal"
        if vol_ratio <= float(config.low_vol_threshold):
            volatility_regime = "low"
        elif vol_ratio >= float(config.high_vol_threshold):
            volatility_regime = "high"

        trend_regime = "flat"
        if trend_z >= float(config.trend_threshold):
            trend_regime = "up"
        elif trend_z <= -float(config.trend_threshold):
            trend_regime = "down"

        dispersion_regime = "high" if dispersion_ratio >= float(config.high_dispersion_threshold) else "low"
        market_regime_key = f"{volatility_regime}|{trend_regime}|{dispersion_regime}"
        regime_label = f"{volatility_regime}_vol/{trend_regime}_trend/{dispersion_regime}_disp"
        confidence = self._confidence(
            vol_ratio=vol_ratio,
            trend_z=trend_z,
            dispersion_ratio=dispersion_ratio,
            config=config,
        )
        features = {
            "short_window": short_window,
            "long_window": long_window,
            "market_return_proxy_mean": short_mean,
            "short_vol": short_vol,
            "long_vol": long_vol,
            "vol_ratio": vol_ratio,
            "trend_z": trend_z,
            "short_dispersion": short_dispersion_mean,
            "long_dispersion": long_dispersion_mean,
            "dispersion_ratio": dispersion_ratio,
        }
        return _HeuristicRegimeState(
            market_regime_key=market_regime_key,
            regime_label=regime_label,
            confidence=confidence,
            features=features,
        )

    def _learned_regime(
        self,
        *,
        market_proxy: pd.Series,
        cross_sectional_dispersion: pd.Series,
        returns_frame: pd.DataFrame,
        config: LearnedRegimeConfig,
    ) -> dict[str, float | int | str] | None:
        if (
            not config.enabled
            or not SKLEARN_AVAILABLE
            or not TSFRESH_AVAILABLE
            or len(market_proxy) < int(config.feature_window + config.min_train_windows)
        ):
            return None
        window_end_positions = list(range(int(config.feature_window) - 1, len(market_proxy)))
        if len(window_end_positions) <= int(config.min_train_windows):
            return None
        current_end = window_end_positions[-1]
        train_end_positions = window_end_positions[:-1]
        if int(config.history_window) > 0 and len(train_end_positions) > int(config.history_window):
            train_end_positions = train_end_positions[-int(config.history_window) :]
        if len(train_end_positions) < int(config.min_train_windows):
            return None
        all_window_positions = [*train_end_positions, current_end]
        manual_features = [
            self._manual_window_features(
                end_position=end_position,
                feature_window=int(config.feature_window),
                market_proxy=market_proxy,
                cross_sectional_dispersion=cross_sectional_dispersion,
                returns_frame=returns_frame,
            )
            for end_position in all_window_positions
        ]
        manual_frame = pd.DataFrame(manual_features).replace([np.inf, -np.inf], np.nan)
        tsfresh_frame = self._tsfresh_window_features(
            market_proxy=market_proxy,
            cross_sectional_dispersion=cross_sectional_dispersion,
            window_end_positions=all_window_positions,
            feature_window=int(config.feature_window),
        )
        feature_frame = pd.concat([manual_frame.reset_index(drop=True), tsfresh_frame.reset_index(drop=True)], axis=1)
        train_frame = feature_frame.iloc[:-1].copy()
        current_frame = feature_frame.iloc[[-1]].copy()
        selected_columns = [
            column
            for column in train_frame.columns
            if train_frame[column].notna().sum() > 1 and train_frame[column].nunique(dropna=True) > 1
        ]
        if not selected_columns:
            return None
        train_selected = train_frame.loc[:, selected_columns].copy()
        current_selected = current_frame.loc[:, selected_columns].copy()
        medians = train_selected.median(numeric_only=True).fillna(0.0)
        train_selected = train_selected.fillna(medians).fillna(0.0)
        current_selected = current_selected.fillna(medians).fillna(0.0)
        cluster_count = min(int(config.cluster_count), len(train_selected))
        if cluster_count <= 1:
            return None
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_selected)
        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=7,
            batch_size=max(32, min(256, len(train_selected))),
            n_init="auto",
        )
        model.fit(scaled_train)
        scaled_current = scaler.transform(current_selected)
        distances = model.transform(scaled_current)[0]
        cluster_id = int(np.argmin(distances))
        confidence = self._cluster_confidence(distances)
        current_features = {str(column): float(current_selected.iloc[0][column]) for column in selected_columns}
        return {
            "learned_cluster_id": cluster_id,
            "learned_confidence": float(confidence),
            "learned_train_window_count": int(len(train_selected)),
            "learned_feature_count": int(len(selected_columns)),
            "learned_model_type": str(config.model_type),
            "learned_tsfresh_profile": str(config.tsfresh_profile),
            **current_features,
        }

    def _manual_window_features(
        self,
        *,
        end_position: int,
        feature_window: int,
        market_proxy: pd.Series,
        cross_sectional_dispersion: pd.Series,
        returns_frame: pd.DataFrame,
    ) -> dict[str, float]:
        start_position = max(0, end_position - feature_window + 1)
        market_window = market_proxy.iloc[start_position : end_position + 1].dropna()
        dispersion_window = cross_sectional_dispersion.iloc[start_position : end_position + 1].dropna()
        returns_window = returns_frame.iloc[start_position : end_position + 1]
        breadth_series = returns_window.gt(0).mean(axis=1).dropna()
        market_cumulative = (1.0 + market_window.fillna(0.0)).cumprod()
        running_max = market_cumulative.cummax()
        drawdown = ((market_cumulative / running_max) - 1.0).min() if not market_cumulative.empty else 0.0
        return {
            "ta_market_realized_vol_5": self._tail_std(market_window, 5),
            "ta_market_realized_vol_20": self._tail_std(market_window, 20),
            "ta_market_realized_vol_63": self._tail_std(market_window, 63),
            "ta_dispersion_realized_vol_20": self._tail_std(dispersion_window, 20),
            "ta_market_momentum_5": self._tail_sum(market_window, 5),
            "ta_market_momentum_20": self._tail_sum(market_window, 20),
            "ta_market_momentum_63": self._tail_sum(market_window, 63),
            "ta_market_drawdown_63": float(drawdown or 0.0),
            "ta_market_autocorr_1": self._autocorr(market_window, lag=1),
            "ta_market_autocorr_5": self._autocorr(market_window, lag=5),
            "ta_dispersion_autocorr_1": self._autocorr(dispersion_window, lag=1),
            "ta_market_skew_63": float(market_window.skew() or 0.0),
            "ta_market_kurtosis_63": float(market_window.kurtosis() or 0.0),
            "ta_dispersion_skew_63": float(dispersion_window.skew() or 0.0),
            "ta_dispersion_kurtosis_63": float(dispersion_window.kurtosis() or 0.0),
            "ta_dispersion_trend_20": float((dispersion_window.tail(5).mean() or 0.0) - (dispersion_window.head(5).mean() or 0.0)),
            "ta_breadth_positive_rate_20": float(breadth_series.tail(20).mean() or 0.0),
            "ta_breadth_positive_vol_20": float(breadth_series.tail(20).std(ddof=0) or 0.0),
        }

    def _tsfresh_window_features(
        self,
        *,
        market_proxy: pd.Series,
        cross_sectional_dispersion: pd.Series,
        window_end_positions: list[int],
        feature_window: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for window_id, end_position in enumerate(window_end_positions):
            start_position = max(0, end_position - feature_window + 1)
            market_window = market_proxy.iloc[start_position : end_position + 1].reset_index(drop=True)
            dispersion_window = cross_sectional_dispersion.iloc[start_position : end_position + 1].reset_index(drop=True)
            for index, value in enumerate(market_window.tolist()):
                rows.append(
                    {
                        "id": window_id,
                        "time": index,
                        "kind": "market_proxy",
                        "value": float(value or 0.0),
                    }
                )
            for index, value in enumerate(dispersion_window.tolist()):
                rows.append(
                    {
                        "id": window_id,
                        "time": index,
                        "kind": "dispersion",
                        "value": float(value or 0.0),
                    }
                )
        if not rows:
            return pd.DataFrame()
        payload = pd.DataFrame(rows)
        extracted = extract_features(
            payload,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=TSFRESH_PARAMETERS,
            disable_progressbar=True,
            n_jobs=0,
        )
        extracted = extracted.replace([np.inf, -np.inf], np.nan)
        extracted.columns = [f"tsfresh_{column}" for column in extracted.columns]
        return extracted

    def _fallback(self, *, region: str, legacy_regime_key: str, global_regime_key: str) -> RegimeSnapshot:
        return RegimeSnapshot(
            region=region,
            legacy_regime_key=legacy_regime_key,
            global_regime_key=global_regime_key,
            market_regime_key="",
            effective_regime_key=legacy_regime_key,
            regime_label="unknown",
            confidence=0.0,
            features={},
        )

    def _confidence(
        self,
        *,
        vol_ratio: float,
        trend_z: float,
        dispersion_ratio: float,
        config: RegimeDetectionConfig,
    ) -> float:
        vol_strength = max(
            abs(vol_ratio - float(config.low_vol_threshold)),
            abs(vol_ratio - float(config.high_vol_threshold)),
        )
        trend_strength = abs(trend_z)
        dispersion_strength = abs(dispersion_ratio - 1.0)
        score = 0.35 * min(1.0, vol_strength) + 0.40 * min(1.0, trend_strength) + 0.25 * min(1.0, dispersion_strength)
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _cluster_confidence(distances: np.ndarray) -> float:
        if len(distances) == 0:
            return 0.0
        ordered = np.sort(np.asarray(distances, dtype=float))
        if len(ordered) == 1:
            return float(max(0.0, min(1.0, 1.0 / (1.0 + ordered[0]))))
        nearest = float(ordered[0])
        second = float(ordered[1])
        return float(max(0.0, min(1.0, 1.0 - (nearest / max(second, 1e-9)))))

    @staticmethod
    def _tail_std(series: pd.Series, window: int) -> float:
        return float(series.tail(window).std(ddof=0) or 0.0)

    @staticmethod
    def _tail_sum(series: pd.Series, window: int) -> float:
        return float(series.tail(window).sum() or 0.0)

    @staticmethod
    def _autocorr(series: pd.Series, *, lag: int) -> float:
        if len(series) <= lag:
            return 0.0
        value = series.autocorr(lag=lag)
        return float(0.0 if pd.isna(value) else value)
