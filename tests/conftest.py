import pytest


@pytest.fixture(autouse=True, scope="session")
def pystorms_cache(tmp_path_factory):
    """Keep derived networks and SWMM run files out of the user's cache."""
    cache = tmp_path_factory.mktemp("pystorms_cache")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PYSTORMS_CACHE", str(cache))
    yield cache
    monkeypatch.undo()
