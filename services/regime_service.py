from __future__ import annotations

import hashlib
import json
import math

import pandas as pd

from core.config import RegimeDetectionConfig
from services.models import RegimeSnapshot


class RegimeService:
    """Infer a lightweight market-state regime without replacing legacy scope keys."""

    def resolve(
        self,
        *,
        matrices,
        config: RegimeDetectionConfig,
        region: str,
        legacy_regime_key: str,
        global_regime_key: str,
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

        short_window = int(max(2, config.short_window))
        long_window = int(max(short_window, config.long_window))
        common_index = market_proxy.index.intersection(cross_sectional_dispersion.index)
        market_proxy = market_proxy.loc[common_index]
        cross_sectional_dispersion = cross_sectional_dispersion.loc[common_index]
        if len(common_index) < int(config.min_points):
            return self._fallback(region=region, legacy_regime_key=legacy_regime_key, global_regime_key=global_regime_key)

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
