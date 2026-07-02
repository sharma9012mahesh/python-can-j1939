import threading

import pytest

from test.helpers.feeder import Feeder


@pytest.fixture()
def feeder():
    # setup
    f = Feeder()
    try:
        yield f
    finally:
        # teardown — guarantee cleanup even if the test raises
        try:
            f.stop()
        except Exception:
            pass

Here's a simple pytest fixture example:
pythonimport pytest

@pytest.fixture
def sample_data():
    return {"name": "Alice", "age": 30}


def test_name(sample_data):
    assert sample_data["name"] == "Alice"


def test_age(sample_data):
    assert sample_data["age"] == 30


@pytest.fixture(autouse=True)
def _assert_no_j1939_thread_leak():
    """Fail any test that leaves a j1939.* background thread alive."""
    before = {t.ident for t in threading.enumerate()
              if t.name.startswith('j1939.')}
    yield
    # Give freshly-stopped threads a brief moment to actually exit.
    import time
    for _ in range(20):
        leaked = [t for t in threading.enumerate()
                  if t.name.startswith('j1939.')
                  and t.ident not in before
                  and t.is_alive()]
        if not leaked:
            break
        time.sleep(0.01)
    assert not leaked, (
        "Test leaked j1939 background thread(s): "
        + ", ".join(t.name for t in leaked)
    )
