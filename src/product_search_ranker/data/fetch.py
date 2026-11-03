"""Download the ESCI files to data/raw/ and record them in data/MANIFEST.json."""

import hashlib
import json
import time
from datetime import date
from pathlib import Path

import httpx

from product_search_ranker import config

LFS_MAGIC = b"version https://git-lfs"
CHUNK = 1 << 20
MAX_ATTEMPTS = 6


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _remote_size(client: httpx.Client, url: str) -> int | None:
    r = client.head(url, follow_redirects=True)
    r.raise_for_status()
    n = r.headers.get("content-length")
    return int(n) if n else None


def _download(client: httpx.Client, url: str, dest: Path, size: int | None) -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _download_once(client, url, dest, size)
        except httpx.TransportError as e:
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"{dest.name}: {type(e).__name__}, resuming (attempt {attempt})", flush=True)
            time.sleep(2**attempt)


def _download_once(client: httpx.Client, url: str, dest: Path, size: int | None) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with client.stream("GET", url, headers=headers, follow_redirects=True) as r:
        if have and r.status_code != 206:
            have = 0  # server ignored Range: start over
        r.raise_for_status()
        next_log = 0
        with part.open("ab" if have else "wb") as f:
            for block in r.iter_bytes(CHUNK):
                f.write(block)
                have += len(block)
                if have >= next_log:
                    total = f"/{size / 1e6:.0f}" if size else ""
                    print(f"{dest.name}: {have / 1e6:.0f}{total} MB", flush=True)
                    next_log = have + 50_000_000
    with part.open("rb") as f:
        if f.read(len(LFS_MAGIC)) == LFS_MAGIC:
            part.unlink()
            raise RuntimeError(f"{url} returned a Git LFS pointer, not the data")
    part.replace(dest)


def fetch(if_changed: bool = False) -> Path:
    raw = config.raw_dir()
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw.parent / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with httpx.Client(timeout=httpx.Timeout(60.0, read=300.0)) as client:
        for name, host in config.ESCI_FILES.items():
            url = f"{config.ESCI_BASES[host]}/{name}"
            dest = raw / name
            size = _remote_size(client, url)
            entry = manifest.get(name)
            done = dest.exists() and entry and size in (None, dest.stat().st_size)
            if done and (not if_changed or _sha256(dest) == entry["sha256"]):
                print(f"{name}: up to date")
                continue
            if dest.exists():
                dest.unlink()
            _download(client, url, dest, size)
            manifest[name] = {
                "url": url,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "downloaded": date.today().isoformat(),
                "source": "github-lfs" if host == "media" else "github-raw",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")
    return manifest_path
