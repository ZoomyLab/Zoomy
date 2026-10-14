"""Backend end-to-end tests are opt-in (`-m backend_e2e`). One container per backend per session,
started on first use and torn down at the end; results go to `$E2E_WORK/report-<checkout>.json`."""
import json
import signal
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402


def pytest_configure(config):
    """A TERM (e.g. `timeout`) unwinds like Ctrl-C, so the containers are torn down."""
    def interrupt(*_):
        raise KeyboardInterrupt
    if "backend_e2e" in (config.option.markexpr or ""):
        signal.signal(signal.SIGTERM, interrupt)


def pytest_collection_modifyitems(config, items):
    if "backend_e2e" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="backend e2e tests are opt-in: pytest tests/backend_e2e -m backend_e2e")
    for item in items:
        if "backend_e2e" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def snap():
    """A snapshot of E2E_CHECKOUT: the tested library (dev mode) and the matrix definitions."""
    return H.snapshot()


@pytest.fixture(scope="session")
def lib(snap):
    """The library the GUI side (composition, engine.py, in-process runs) uses."""
    return snap / "library"


@pytest.fixture(scope="session")
def report():
    rows = []
    yield rows
    H.WORK.mkdir(parents=True, exist_ok=True)
    out = H.WORK / f"report-{H.CHECKOUT.name}-{H.MODE}.json"
    old = json.loads(out.read_text()) if out.exists() else {"rows": []}
    keep = {(r["test"]) for r in rows}
    merged = [r for r in old["rows"] if r["test"] not in keep] + rows
    out.write_text(json.dumps({"checkout": str(H.CHECKOUT), "mode": H.MODE,
                               "provenance": H.provenance(), "rows": merged}, indent=1))


#: where every server sees the shared coupling directory (zoomy_server.peers: ZOOMY_COUPLE_DIR)
#: and the registry of the servers up (ZOOMY_PEERS, `{adapter tag: url}`, rewritten as servers
#: come up): a coupled case's job submits its other participants to their backends' servers
COUPLE = "/e2e/couple"


@pytest.fixture(scope="session")
def servers(snap):
    """`servers(name) -> url`: the backend's zoomy-server, started once (skip without image).
    Every server binds one coupling directory at `COUPLE` and finds the others in its
    `peers.json`."""
    up, started = {}, []
    couple = H.WORK / f"{H.CHECKOUT.name}-{H.MODE}" / "couple"
    couple.mkdir(parents=True, exist_ok=True)
    registry = couple / "peers.json"
    registry.write_text("{}")

    def get(name):
        if name not in up:
            b = H.BACKENDS[name]
            img, how = H.image(b)
            if img is None:
                up[name] = ("skip", f"{name}: {how}")
            else:
                c = H.Container(b, img, snap, H.WORK / f"{H.CHECKOUT.name}-{H.MODE}" / name,
                                dev=H.MODE == "dev", shared=((couple, COUPLE),),
                                extra_env={"ZOOMY_COUPLE_DIR": COUPLE,
                                           "ZOOMY_PEERS": f"{COUPLE}/peers.json"})
                started.append(c)
                t0 = time.monotonic()
                try:
                    url = c.serve()
                    up[name] = ("url", url, how, round(time.monotonic() - t0, 1), c)
                    peers = json.loads(registry.read_text())
                    peers[b.adapter] = url
                    registry.write_text(json.dumps(peers))
                except Exception as e:
                    up[name] = ("fail", f"{name}: server did not come up: {e}")
        kind, val = up[name][:2]
        if kind == "skip":
            pytest.skip(val)
        if kind == "fail":
            pytest.fail(val, pytrace=False)
        return val

    get.info = up

    yield get
    for c in started:
        c.stop()
