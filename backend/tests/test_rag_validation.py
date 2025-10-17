from types import SimpleNamespace

from app.services.rag import _apply_numeric_validation


def test_apply_numeric_validation_success() -> None:
    table_lookup = {
        "table-1": SimpleNamespace(rows=[{"Metric": "Revenue", "2022": "1,000", "2023": "1,200"}])
    }

    content, suffix = _apply_numeric_validation("Revenue was 1,000 in 2022.", table_lookup)

    assert "Validation:" in content
    assert suffix is not None
    assert "unable to verify" not in content


def test_apply_numeric_validation_warns_when_missing() -> None:
    table_lookup = {
        "table-1": SimpleNamespace(rows=[{"Metric": "Revenue", "2022": "1,000"}])
    }

    content, suffix = _apply_numeric_validation("Revenue was 9,999 in 2022.", table_lookup)

    assert "unable to verify" in content
    assert suffix is not None
