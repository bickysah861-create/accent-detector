"""Minimal seekable file-like object over HTTP range requests.

Lets pyarrow read a remote parquet footer + selected column chunks without
pulling the whole shard. The dataset is 8.85 GB; we only need a few hundred MB.
"""
from __future__ import annotations

import requests


class HttpRangeFile:
    def __init__(self, url: str, session: requests.Session | None = None, timeout: int = 60):
        self.url = url
        self.timeout = timeout
        self.session = session or requests.Session()
        self._pos = 0
        self._closed = False
        self._size: int | None = None
        # Resolve redirects once so every range request hits the final CDN URL.
        r = self.session.head(url, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        self.url = r.url
        self._size = int(r.headers["Content-Length"])

    # -- pyarrow NativeFile protocol -------------------------------------
    def size(self) -> int:
        return self._size

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self._size + pos
        return self._pos

    def read(self, nbytes: int | None = None) -> bytes:
        if nbytes is None:
            nbytes = self._size - self._pos
        if nbytes <= 0:
            return b""
        start = self._pos
        end = min(start + nbytes, self._size) - 1
        if end < start:
            return b""
        data = self._get_range(start, end)
        self._pos = start + len(data)
        return data

    def _get_range(self, start: int, end: int, attempts: int = 4) -> bytes:
        headers = {"Range": f"bytes={start}-{end}"}
        last = None
        for i in range(attempts):
            try:
                r = self.session.get(self.url, headers=headers, timeout=self.timeout)
                if r.status_code in (200, 206):
                    return r.content
                last = RuntimeError(f"HTTP {r.status_code}")
            except Exception as e:  # transient network faults are common here
                last = e
        raise last

    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def flush(self): pass

    # `closed` must be a property: pyarrow tests it as an attribute, and a
    # bound method is always truthy -> the file would look permanently closed.
    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True
