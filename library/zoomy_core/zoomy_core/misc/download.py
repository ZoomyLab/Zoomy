"""Fetch a file by URL with one synchronous call in CPython and in Pyodide."""

import os
import sys


def download(urls, path):
    """Write the first of `urls` (one URL or several, tried in order) that answers to `path`.

    - An existing `path` is kept.
    - CPython reads with `urllib`. Pyodide cannot open sockets, so it uses a synchronous
      `XMLHttpRequest` (allowed in a Web Worker, where the GUI and its notebook kernel
      run Python); a cross-origin URL must then send CORS headers.
    """
    if os.path.exists(path):
        return path
    errors = []
    for url in [urls] if isinstance(urls, str) else list(urls):
        try:
            data = _get(url)
        except Exception as e:
            errors.append(f"{url}: {e}")
            continue
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
        return path
    raise OSError("download failed:\n  " + "\n  ".join(errors))


def _get(url):
    if sys.platform == "emscripten":
        from js import XMLHttpRequest

        req = XMLHttpRequest.new()
        req.open("GET", url, False)
        req.responseType = "arraybuffer"
        req.send(None)
        if not 200 <= req.status < 300:
            raise OSError(f"HTTP {req.status}")
        return req.response.to_bytes()
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "zoomy_core"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()
