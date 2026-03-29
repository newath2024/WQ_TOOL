from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_brain_operator_catalog_accepts_expanded_export_json_without_pdf(tmp_path: Path) -> None:
    export_path = tmp_path / "worldquant_brain_operators_expanded.json"
    output_path = tmp_path / "brain_operator_catalog.json"
    export_path.write_text(
        json.dumps(
            {
                "exported_at": "2026-03-29T10:24:52.164Z",
                "page_url": "https://platform.worldquantbrain.com/learn/operators",
                "title": "Operators",
                "operator_count": 2,
                "operators": [
                    {
                        "signature": "ts_mean(x, d)",
                        "name": "ts_mean",
                        "tier": "base",
                        "scope": "Combo, Regular",
                        "summary": "Rolling mean.",
                        "detail_text": "Uses the last d days of x. Examples: ts_mean(close, 20). Tip: use a longer lookback to smooth turnover.",
                        "detail_html": "<p>Uses the last d days of x.</p><p><b>Examples:</b></p><p>ts_mean(close, 20)</p><p><b>Tip:</b> use a longer lookback to smooth turnover.</p>",
                        "images": [{"src": "blob:chart", "alt": "example chart", "width": 640, "height": 480}],
                    },
                    {
                        "signature": "group_rank(x, group)",
                        "name": "group_rank",
                        "tier": "expert",
                        "scope": "Regular",
                        "summary": "Ranks within each group.",
                        "detail_text": "Useful for sector-relative ranking.",
                        "detail_html": "",
                        "images": [],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "tools/build_brain_operator_catalog.py",
            "--export-json",
            str(export_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["source_pdf"] is None
    assert payload["source_export_json"] == str(export_path)
    assert payload["operator_count"] == 2
    assert payload["operators"][0]["signature"] == "ts_mean(x, d)"
    assert "Uses the last d days of x." in payload["operators"][0]["details"]
    assert "<p>Uses the last d days of x.</p>" in payload["operators"][0]["detail_html"]
    assert payload["operators"][0]["images"][0]["alt"] == "example chart"
    assert "ts_mean(close, 20)" in payload["operators"][0]["examples"]
    assert payload["operators"][0]["tips"]
    assert "smoothing" in payload["operators"][0]["behavior_tags"]
    assert payload["operators"][1]["tier"] == "expert"
