from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from core.config import RegimeDetectionConfig
from services.regime_service import RegimeService


def test_regime_service_produces_stable_market_key_when_data_is_sufficient() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01 + ((index % 5) - 2) * 0.002 for index in range(80)],
            "BBB": [0.012 + ((index % 7) - 3) * 0.003 for index in range(80)],
            "CCC": [0.008 + ((index % 3) - 1) * 0.001 for index in range(80)],
        }
    )
    matrices = SimpleNamespace(numeric_fields={"returns": returns})

    snapshot = RegimeService().resolve(
        matrices=matrices,
        config=RegimeDetectionConfig(),
        region="USA",
        legacy_regime_key="legacy-key",
        global_regime_key="global-key",
    )

    assert snapshot.market_regime_key
    assert snapshot.effective_regime_key != "legacy-key"
    assert snapshot.confidence >= 0.0


def test_regime_service_falls_back_when_data_is_insufficient() -> None:
    returns = pd.DataFrame({"AAA": [0.01, 0.02], "BBB": [0.0, 0.01]})
    matrices = SimpleNamespace(numeric_fields={"returns": returns})

    snapshot = RegimeService().resolve(
        matrices=matrices,
        config=RegimeDetectionConfig(min_points=20),
        region="USA",
        legacy_regime_key="legacy-key",
        global_regime_key="global-key",
    )

    assert snapshot.market_regime_key == ""
    assert snapshot.effective_regime_key == "legacy-key"
    assert snapshot.regime_label == "unknown"
