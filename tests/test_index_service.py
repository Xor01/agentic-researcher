from app.services import index_service


def test_query_engine_is_built_once_and_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(index_service, "_query_engine", None)
    monkeypatch.setattr(index_service, "_build_query_engine", lambda: calls.append(1) or object())

    first = index_service.load_query_engine()
    second = index_service.load_query_engine()

    assert first is second
    assert len(calls) == 1
