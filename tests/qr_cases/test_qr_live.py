"""Live link check (run after a deploy): every QR target answers 200 on the deployed site,
every project zip it opens is served with the repo's file list (`-m qr_live`); the GUI
serves the wheels built from this commit (`gui/gui/wheels/`) and the data the sessions fetch
from the site (`-m qr_deploy`: only true right after this commit's deploy).

The fragment of a `#/open?...` URL never reaches the server, so the page is checked plus the
zip it names. Files are fetched with a cache-busting query and retried while the Pages CDN may
still serve the previous deploy.
"""
import hashlib
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qr_manifest as Q  # noqa: E402

TARGETS = Q.targets()
ZIPS = sorted({(t.zip_url, t.local_zip) for t in TARGETS if t.project}, key=lambda z: z[0])
TRIES, WAIT = 6, 30


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "zoomy-qr-live-check"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read()


@pytest.mark.qr_live
@pytest.mark.parametrize("target", TARGETS, ids=[t.key for t in TARGETS])
def test_qr_target_live(target):
    for attempt in range(TRIES):
        try:
            status, _ = _get(target.url)
            break
        except urllib.error.HTTPError as e:
            status = e.code
        except urllib.error.URLError as e:
            status = f"unreachable ({e.reason})"
        time.sleep(WAIT if attempt + 1 < TRIES else 0)
    assert status == 200, f"{target.key}: {target.url} -> {status}"


@pytest.mark.qr_live
@pytest.mark.parametrize("zip_url,local", ZIPS, ids=[Path(z).name for z, _ in ZIPS])
def test_project_zip_live(zip_url, local):
    want = sorted(zipfile.ZipFile(local).namelist())
    got = None
    for attempt in range(TRIES):
        try:
            status, body = _get(f"{zip_url}?qr={int(time.time())}")
            got = sorted(zipfile.ZipFile(io.BytesIO(body)).namelist())
            if got == want:
                return
        except urllib.error.HTTPError as e:
            status = e.code
        except (urllib.error.URLError, zipfile.BadZipFile) as e:
            status = repr(e)
        time.sleep(WAIT if attempt + 1 < TRIES else 0)
    assert status == 200, f"{zip_url} -> {status}"
    assert got == want, (f"{zip_url}: deployed file list differs from {local.relative_to(Q.ROOT)}\n"
                         f"  only live: {sorted(set(got) - set(want))}\n"
                         f"  only repo: {sorted(set(want) - set(got))}")


GUI = Q.SITE + "gui/gui/"
#: Files the deploy mirrors from release assets into gui/gui/data/ (render-webpage.yml),
#: so the browser may read them (GitHub release downloads send no CORS headers).
MIRRORED = ["sme_vof_snap_gui_l1.tar.gz"]


def _get_fresh(url, check):
    """GET `url` until `check(body)` holds (or TRIES run out); return the last body."""
    body = None
    for attempt in range(TRIES):
        try:
            _, body = _get(f"{url}?qr={int(time.time())}")
            if check(body):
                return body
        except urllib.error.URLError as e:
            body = repr(e)
        time.sleep(WAIT if attempt + 1 < TRIES else 0)
    return body


def _commit():
    """The checkout's commit: zoomy_core and zoomy_plotting are folders of this repository."""
    return subprocess.check_output(["git", "-C", str(Q.ROOT), "rev-parse", "HEAD"], text=True).strip()


@pytest.mark.qr_deploy
def test_gui_wheels_live():
    """gui/gui/wheels/index.json names wheels of this commit, served with their sha256."""
    want = {"zoomy-core": _commit(), "zoomy-plotting": _commit()}

    def current(body):
        try:
            return {k: v["commit"] for k, v in json.loads(body).items()} == want
        except (TypeError, ValueError, KeyError):
            return False

    body = _get_fresh(GUI + "wheels/index.json", current)
    assert current(body), f"{GUI}wheels/index.json: {body!r:.400} (want commits {want})"
    for name, e in json.loads(body).items():
        whl = _get_fresh(GUI + "wheels/" + e["wheel"],
                         lambda b: hashlib.sha256(b).hexdigest() == e["sha256"])
        assert isinstance(whl, bytes) and hashlib.sha256(whl).hexdigest() == e["sha256"], \
            f"{name}: {e['wheel']} not served as built"


@pytest.mark.qr_deploy
@pytest.mark.parametrize("name", MIRRORED)
def test_mirrored_data_live(name):
    body = _get_fresh(GUI + "data/" + name, lambda b: isinstance(b, bytes) and len(b) > 0)
    assert isinstance(body, bytes) and body, f"{GUI}data/{name}: {body!r:.200}"
