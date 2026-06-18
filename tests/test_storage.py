from datetime import datetime, timedelta, timezone

import pytest

from detection.risk_score import RiskScore
from detection.storage import (
    get_feature_vector,
    get_latest_scores,
    get_shap_values,
    init_db,
    save_feature_vectors,
    save_scores,
    save_shap_values,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "ledgerlens.db")


def _score(wallet="GABC", asset_pair="XLM/USDC", score=80, timestamp=None) -> RiskScore:
    return RiskScore(
        wallet=wallet,
        asset_pair=asset_pair,
        score=score,
        benford_flag=score > 50,
        ml_flag=score > 50,
        confidence=90,
        timestamp=timestamp or datetime.now(timezone.utc),
    )


def test_init_db_creates_table(db_path):
    init_db(db_path)
    assert get_latest_scores(db_path=db_path) == []


def test_save_and_get_latest_scores(db_path):
    save_scores([_score()], db_path)
    scores = get_latest_scores(db_path=db_path)
    assert len(scores) == 1
    assert scores[0].wallet == "GABC"
    assert scores[0].score == 80


def test_get_latest_scores_returns_most_recent_per_wallet_asset_pair(db_path):
    older = _score(score=30, timestamp=datetime.now(timezone.utc) - timedelta(hours=1))
    newer = _score(score=90, timestamp=datetime.now(timezone.utc))
    save_scores([older, newer], db_path)

    scores = get_latest_scores(db_path=db_path)
    assert len(scores) == 1
    assert scores[0].score == 90


def test_get_latest_scores_filters_by_wallet(db_path):
    save_scores([_score(wallet="GABC"), _score(wallet="GXYZ")], db_path)

    scores = get_latest_scores(wallet="GXYZ", db_path=db_path)
    assert len(scores) == 1
    assert scores[0].wallet == "GXYZ"


def test_get_latest_scores_filters_flags_in_sql(monkeypatch):
    executed = []

    class FakeCursor:
        def fetchall(self):
            return []

    class FakeConnection:
        def executescript(self, _script):
            return None

        def commit(self):
            return None

        def execute(self, query, params):
            executed.append((query, params))
            return FakeCursor()

    from contextlib import contextmanager

    @contextmanager
    def fake_connect(_db_path=None):
        yield FakeConnection()

    monkeypatch.setattr("detection.storage._connect", fake_connect)

    get_latest_scores(benford_flag=True, ml_flag=False, db_path="fake.db")

    query, params = executed[-1]
    compact_query = " ".join(query.split())
    assert "rs.benford_flag = ?" in compact_query
    assert "rs.ml_flag = ?" in compact_query
    assert params == (1, 0)


def test_get_latest_scores_sorts_by_requested_column_in_sql(monkeypatch):
    executed = []

    class FakeCursor:
        def fetchall(self):
            return []

    class FakeConnection:
        def executescript(self, _script):
            return None

        def commit(self):
            return None

        def execute(self, query, params):
            executed.append((query, params))
            return FakeCursor()

    from contextlib import contextmanager

    @contextmanager
    def fake_connect(_db_path=None):
        yield FakeConnection()

    monkeypatch.setattr("detection.storage._connect", fake_connect)

    get_latest_scores(sort_by="confidence", db_path="fake.db")

    query, _params = executed[-1]
    assert "ORDER BY rs.confidence DESC" in " ".join(query.split())


def test_get_latest_scores_rejects_invalid_sort_by(db_path):
    with pytest.raises(ValueError, match="sort_by"):
        get_latest_scores(sort_by="invalid", db_path=db_path)


def test_save_scores_noop_on_empty_list(db_path):
    save_scores([], db_path)
    assert get_latest_scores(db_path=db_path) == []


def test_get_latest_scores_applies_limit_offset_in_sql(tmp_path, monkeypatch):
    """Ensure paging is done in SQL, not by loading all rows in Python."""
    import detection.storage as storage_module

    db_path = str(tmp_path / "ledgerlens.db")

    # Mock sqlite3 connection and cursor behavior
    calls = {}

    class FakeConn:
        def __init__(self):
            self._executed = []

        def execute(self, query, params):
            calls["query"] = query
            calls["params"] = params

            class FakeCursor:
                def fetchall(self_inner):
                    return []

            return FakeCursor()

        def executescript(self, _):
            return None

        def commit(self):
            return None

        def close(self):
            return None

    class FakeContext:
        def __enter__(self_inner):
            return FakeConn()

        def __exit__(self_inner, exc_type, exc, tb):
            return False

    def fake_connect(_db_path=None):
        return FakeContext()

    monkeypatch.setattr(storage_module, "_connect", lambda db_path=None: fake_connect(db_path))

    storage_module.init_db(db_path)
    storage_module.get_latest_scores(wallet=None, limit=5, offset=10, db_path=db_path)

    assert "LIMIT ? OFFSET ?" in calls["query"]
    assert calls["params"] == (5, 10)


# ---------------------------------------------------------------------------
# Feature vector + SHAP value round-trip tests
# ---------------------------------------------------------------------------

_FEATURES = {
    "benford_mad_24h": 0.02,
    "round_trip_trade_frequency": 0.1,
    "network_centrality": 0.5,
}


def test_save_and_get_feature_vector(db_path):
    save_feature_vectors(
        [{"wallet": "GABC", "asset_pair": "XLM/USDC", "features": _FEATURES}],
        db_path,
    )
    result = get_feature_vector("GABC", "XLM/USDC", db_path)
    assert result is not None
    assert result["benford_mad_24h"] == pytest.approx(0.02)
    assert result["round_trip_trade_frequency"] == pytest.approx(0.1)


def test_get_feature_vector_returns_none_for_unknown_wallet(db_path):
    init_db(db_path)
    assert get_feature_vector("GUNKNOWN", "XLM/USDC", db_path) is None


def test_get_feature_vector_returns_most_recent(db_path):
    save_feature_vectors(
        [{"wallet": "GABC", "asset_pair": "XLM/USDC", "features": {"benford_mad_24h": 0.01}}],
        db_path,
    )
    save_feature_vectors(
        [{"wallet": "GABC", "asset_pair": "XLM/USDC", "features": {"benford_mad_24h": 0.99}}],
        db_path,
    )
    result = get_feature_vector("GABC", "XLM/USDC", db_path)
    assert result["benford_mad_24h"] == pytest.approx(0.99)


def test_save_and_get_shap_values(db_path):
    # Must have a feature_vectors row first.
    save_feature_vectors(
        [{"wallet": "GABC", "asset_pair": "XLM/USDC", "features": _FEATURES}],
        db_path,
    )
    shap_payload = [
        {"feature": "benford_mad_24h", "shap_value": 0.35},
        {"feature": "round_trip_trade_frequency", "shap_value": -0.20},
        {"feature": "network_centrality", "shap_value": 0.10},
    ]
    save_shap_values("GABC", "XLM/USDC", shap_payload, db_path)

    result = get_shap_values("GABC", "XLM/USDC", db_path)
    assert result is not None
    assert len(result) == 3
    assert result[0]["feature"] == "benford_mad_24h"
    assert result[0]["shap_value"] == pytest.approx(0.35)


def test_get_shap_values_returns_none_for_unknown_wallet(db_path):
    init_db(db_path)
    assert get_shap_values("GUNKNOWN", "XLM/USDC", db_path) is None


def test_get_shap_values_returns_none_when_not_yet_computed(db_path):
    # Feature vector present but no SHAP values written yet.
    save_feature_vectors(
        [{"wallet": "GABC", "asset_pair": "XLM/USDC", "features": _FEATURES}],
        db_path,
    )
    assert get_shap_values("GABC", "XLM/USDC", db_path) is None


def test_save_feature_vectors_noop_on_empty_list(db_path):
    init_db(db_path)
    save_feature_vectors([], db_path)  # Should not raise.
    assert get_feature_vector("GABC", "XLM/USDC", db_path) is None
