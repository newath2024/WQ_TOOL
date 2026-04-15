from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from core.config import LearnedRegimeConfig, RegimeDetectionConfig
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


def test_regime_service_can_emit_learned_cluster_and_excludes_current_window_from_training() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.0015 * index + ((index % 7) - 3) * 0.0008 for index in range(220)],
            "BBB": [0.0011 * index + ((index % 5) - 2) * 0.0009 for index in range(220)],
            "CCC": [0.0009 * index + ((index % 9) - 4) * 0.0007 for index in range(220)],
        }
    )
    matrices = SimpleNamespace(numeric_fields={"returns": returns})
    learned_config = LearnedRegimeConfig(
        cluster_count=4,
        history_window=80,
        feature_window=20,
        min_train_windows=40,
        confidence_floor=0.0,
    )

    snapshot = RegimeService().resolve(
        matrices=matrices,
        config=RegimeDetectionConfig(min_confidence=0.0),
        region="USA",
        legacy_regime_key="legacy-key",
        global_regime_key="global-key",
        learned_config=learned_config,
    )

    total_windows = len(returns) - learned_config.feature_window + 1
    expected_train_windows = min(learned_config.history_window, total_windows - 1)

    assert snapshot.market_regime_key.startswith("learned_cluster:")
    assert snapshot.features["learned_train_window_count"] == expected_train_windows
    assert snapshot.features["learned_feature_count"] > 0


def test_regime_service_falls_back_to_heuristic_when_learned_confidence_is_too_low(monkeypatch) -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01 + ((index % 5) - 2) * 0.002 for index in range(120)],
            "BBB": [0.012 + ((index % 7) - 3) * 0.003 for index in range(120)],
            "CCC": [0.008 + ((index % 3) - 1) * 0.001 for index in range(120)],
        }
    )
    matrices = SimpleNamespace(numeric_fields={"returns": returns})
    service = RegimeService()
    monkeypatch.setattr(
        service,
        "_learned_regime",
        lambda **kwargs: {"learned_cluster_id": 2, "learned_confidence": 0.1},
    )

    snapshot = service.resolve(
        matrices=matrices,
        config=RegimeDetectionConfig(),
        region="USA",
        legacy_regime_key="legacy-key",
        global_regime_key="global-key",
        learned_config=LearnedRegimeConfig(confidence_floor=0.5),
    )

    assert not snapshot.market_regime_key.startswith("learned_cluster:")
    assert snapshot.features["learned_cluster_id"] == 2
