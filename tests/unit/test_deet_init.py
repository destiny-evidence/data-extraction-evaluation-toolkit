import importlib
import sys


def test_deet_import_eagerly_loads_anyio_submodules(monkeypatch):
    for module_name in ("anyio.lowlevel", "anyio.abc", "anyio", "deet"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    importlib.invalidate_caches()
    importlib.import_module("deet")

    assert "anyio" in sys.modules
    assert "anyio.abc" in sys.modules
    assert "anyio.lowlevel" in sys.modules


def test_deet_import_loads_anyio_submodules_when_anyio_already_present(monkeypatch):
    anyio_module = importlib.import_module("anyio")
    for module_name in ("anyio.lowlevel", "anyio.abc", "anyio", "deet"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "anyio", anyio_module)

    importlib.invalidate_caches()
    importlib.import_module("deet")

    assert "anyio.abc" in sys.modules
    assert "anyio.lowlevel" in sys.modules
