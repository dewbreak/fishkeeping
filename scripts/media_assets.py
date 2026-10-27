"""Explicit, bounded downloads for the opt-in real-media benchmark; never used by CI.

Run ``python scripts/media_assets.py --download`` once, then run offline without the flag.
Only source media lives in .cache/media; the repository stores provenance and frozen labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "real_media_cases.json"
DEFAULT_CACHE = ROOT / ".cache" / "media"
ALLOWED_HOSTS = {"raw.githubusercontent.com", "www.nass.usda.gov", "media.w3.org"}
MAX_ASSET_BYTES = 32 * 1024 * 1024


def load_manifest(path: Path = MANIFEST) -> dict:
    """Load frozen questions, provenance and source hashes without any network access."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(cache_dir: Path, filename: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", filename):
        raise ValueError("Media filename must be a simple basename")
    root = Path(cache_dir).resolve()
    target = root / filename
    if target.is_symlink() or target.resolve().parent != root:
        raise ValueError("Media cache path must stay within its root")
    return target


def _validate_source(asset: dict) -> None:
    parsed = urlsplit(asset["url"])
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ValueError("Only HTTPS media URLs on approved hosts are allowed")
    if not 0 < asset["bytes"] <= asset["max_bytes"] <= MAX_ASSET_BYTES:
        raise ValueError("Invalid media byte budget")
    if not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
        raise ValueError("Source media must have a frozen SHA-256")


def _verify_source(path: Path, asset: dict) -> None:
    if path.stat().st_size != asset["bytes"] or sha256_file(path) != asset["sha256"]:
        raise ValueError(f"Media hash/size mismatch: {path.name}")


def ensure_asset(
    asset_id: str,
    cache_dir: Path = DEFAULT_CACHE,
    *,
    download: bool = False,
    manifest: dict | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Verify cached source, or explicitly download it once with byte and hash bounds.

    Redirects are rejected instead of allowing a trusted URL to redirect to another host.
    Supplying an HTTP client is only a transport injection point for offline unit tests.
    """
    data = load_manifest() if manifest is None else manifest
    asset = data["assets"][asset_id]
    _validate_source(asset)
    target = _cache_path(cache_dir, asset["filename"])
    if target.exists():
        _verify_source(target, asset)
        return target
    if not download:
        raise FileNotFoundError(f"Missing {target.name}; run media_assets.py --download explicitly")
    target.parent.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    active = client or httpx.Client(timeout=60, follow_redirects=False)
    temporary: Path | None = None
    try:
        with active.stream("GET", asset["url"], follow_redirects=False) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length is not None and int(length) > asset["max_bytes"]:
                raise ValueError("Media response exceeds byte budget")
            total = 0
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".media-download-", delete=False
            ) as output:
                temporary = Path(output.name)
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    total += len(chunk)
                    if total > asset["max_bytes"]:
                        raise ValueError("Media stream exceeds byte budget")
                    digest.update(chunk)
                    output.write(chunk)
            if total != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
                raise ValueError("Downloaded media hash/size mismatch")
            os.replace(temporary, target)
            temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if own_client:
            active.close()
    return target


def prepare_case(
    case: dict,
    cache_dir: Path = DEFAULT_CACHE,
    *,
    download: bool = False,
    manifest: dict | None = None,
) -> Path:
    """Return verified image/video input; PDF table cases are rendered by local Poppler.

    Video cases return the identical full source. evidence_intervals are scoring-only;
    explicit question windows are applied downstream. PDF PNG bytes vary across renderer/font
    versions: sources are strictly pinned; actual derived SHA is stored in a sidecar.
    """
    source = ensure_asset(case["asset"], cache_dir, download=download, manifest=manifest)
    render = case.get("render")
    if render is None:
        return source
    target = _cache_path(cache_dir, render["filename"])
    sidecar = _cache_path(cache_dir, render["filename"] + ".provenance.json")
    provenance = {
        "source_sha256": sha256_file(source),
        "page": render["page"],
        "dpi": render["dpi"],
    }
    if target.exists() and sidecar.exists():
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        if any(recorded.get(key) != value for key, value in provenance.items()):
            raise ValueError("Rendered media provenance mismatch")
        if recorded.get("sha256") != sha256_file(target):
            raise ValueError("Rendered media hash mismatch")
        return target
    if target.exists() and sha256_file(target) == render["reference_sha256"]:
        provenance.update(sha256=render["reference_sha256"], renderer=render["reference_renderer"])
        sidecar.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        return target
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("Rendering table fixtures requires local Poppler (pdftoppm)")
    if not 1 <= render["page"] <= 48 or not 72 <= render["dpi"] <= 200:
        raise ValueError("PDF render settings exceed fixture limits")
    version = subprocess.run(
        [executable, "-v"], capture_output=True, text=True, check=True, timeout=15
    )
    with tempfile.TemporaryDirectory(prefix=".media-render-", dir=target.parent) as temporary:
        prefix = Path(temporary) / "page"
        subprocess.run(
            [
                executable,
                "-f",
                str(render["page"]),
                "-l",
                str(render["page"]),
                "-r",
                str(render["dpi"]),
                "-png",
                "-singlefile",
                str(source),
                str(prefix),
            ],
            capture_output=True,
            check=True,
            timeout=60,
        )
        generated = prefix.with_suffix(".png")
        if not generated.is_file() or generated.stat().st_size > MAX_ASSET_BYTES:
            raise ValueError("Invalid rendered page")
        provenance.update(
            sha256=sha256_file(generated),
            renderer=(version.stderr or version.stdout).splitlines()[0],
        )
        os.replace(generated, target)
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true", help="Opt in to bounded source downloads"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    manifest = load_manifest()
    for case in manifest["cases"]:
        path = prepare_case(case, args.cache_dir, download=args.download, manifest=manifest)
        print(json.dumps({"case": case["id"], "path": str(path), "sha256": sha256_file(path)}))


if __name__ == "__main__":
    main()
