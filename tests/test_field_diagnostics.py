from __future__ import annotations

from types import SimpleNamespace

from cli.commands import diagnose_fields
from cli.commands.diagnose_fields import build_field_diagnostic_specs
from cli.app import build_parser
from core.config import load_config
from data.field_registry import FieldSpec


def test_field_diagnostic_builder_emits_expected_expressions_and_limit() -> None:
    field = FieldSpec(
        name="anl69_eps",
        dataset="analyst",
        field_type="matrix",
        coverage=0.8,
        alpha_usage_count=10,
        category="analyst",
        field_score=0.7,
    )

    specs = build_field_diagnostic_specs(
        fields=[field],
        windows=[5, 22],
        bounds=[0.0, 1.0],
        max_expressions=5,
    )

    assert [item["expression"] for item in specs] == [
        "anl69_eps",
        "anl69_eps != 0 ? 1 : -1",
        "ts_std_dev(anl69_eps,5) != 0 ? 1 : -1",
        "ts_std_dev(anl69_eps,22) != 0 ? 1 : -1",
        "abs(anl69_eps) > 0 ? 1 : -1",
    ]


def test_diagnose_fields_dry_run_does_not_initialize_run_or_submit(monkeypatch, capsys) -> None:
    field = FieldSpec(
        name="anl69_eps",
        dataset="analyst",
        field_type="matrix",
        coverage=0.8,
        alpha_usage_count=10,
        category="analyst",
        field_score=0.7,
    )

    class FakeRegistry:
        def generation_numeric_fields(self, allowed_fields, *, include_catalog_fields):
            return [field]

    monkeypatch.setattr(
        diagnose_fields,
        "load_research_context",
        lambda *args, **kwargs: SimpleNamespace(field_registry=FakeRegistry()),
    )
    monkeypatch.setattr(
        diagnose_fields,
        "init_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("init_run called")),
    )

    config = load_config("config/dev.yaml")
    args = SimpleNamespace(top=1, max_expressions=2, windows="5", bounds="0", submit=False)
    environment = SimpleNamespace(context=SimpleNamespace(run_id="run-dry"))

    code = diagnose_fields.handle(args, config, SimpleNamespace(), environment)
    output = capsys.readouterr().out

    assert code == 0
    assert "dry_run: true" in output
    assert "expression_count: 2" in output


def test_diagnose_fields_accepts_explicit_dry_run_flag() -> None:
    args = build_parser().parse_args(["diagnose-fields", "--top", "3", "--dry-run"])

    assert args.command == "diagnose-fields"
    assert args.top == 3
    assert args.dry_run is True
    assert args.submit is False
