"""Live link check (run after a deploy): every QR target answers 200 on the deployed site, and
every project zip it opens is served with the repo's file list.

The fragment of a `#/open?...` URL never reaches the server, so the page is checked plus the
zip it names. Zips are fetched with a cache-busting query and retried while the Pages CDN may
still serve the previous deploy.
"""
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qr_manifest as Q  # noqa: E402

pytestmark = pytest.mark.qr_live
TARGETS = Q.targets()
ZIPS = sorted({(t.zip_url, t.local_zip) for t in TARGETS if t.project}, key=lambda z: z[0])
TRIES, WAIT = 6, 30


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "zoomy-qr-live-check"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read()


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
