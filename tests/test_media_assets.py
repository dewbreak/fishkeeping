"""Media preparation safety tests; every HTTP response is local and deterministic."""

import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "media_assets.py"
SPEC = importlib.util.spec_from_file_location("media_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
media = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(media)


@pytest.fixture
def manifest():
    content = b"licensed test fixture"
    return {
        "assets": {
            "test": {
                "filename": "fixture.png",
                "url": "https://media.w3.org/fixture.png",
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "max_bytes": 32,
            }
        }
    }


def test_missing_cache_requires_explicit_download(tmp_path, manifest):
    def unexpected_request(request):
        pytest.fail("Offline preparation attempted a network request")

    with httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client:
        with pytest.raises(FileNotFoundError, match="--download"):
            media.ensure_asset("test", tmp_path, manifest=manifest, client=client)
    assert list(tmp_path.iterdir()) == []


def test_verified_download_is_reused_offline(tmp_path, manifest):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=b"licensed test fixture")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        path = media.ensure_asset("test", tmp_path, download=True, manifest=manifest, client=client)
        assert media.ensure_asset("test", tmp_path, manifest=manifest, client=client) == path
    assert len(requests) == 1
    assert path.read_bytes() == b"licensed test fixture"
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://media.w3.org/file",
        "file:///tmp/file",
        "https://127.0.0.1/file",
        "https://media.w3.org.evil.test/file",
        "https://user:password@media.w3.org/file",
        "https://media.w3.org:8443/file",
    ],
)
def test_unapproved_source_is_rejected_before_request(tmp_path, manifest, url):
    manifest["assets"]["test"]["url"] = url
    with pytest.raises(ValueError, match="HTTPS"):
        media.ensure_asset("test", tmp_path, download=True, manifest=manifest)


def test_redirect_is_never_followed(tmp_path, manifest):
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        with pytest.raises(httpx.HTTPStatusError):
            media.ensure_asset("test", tmp_path, download=True, manifest=manifest, client=client)
    assert len(requests) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "content,headers",
    [
        (b"X" * 33, {}),
        (b"small", {"content-length": "99999"}),
        (b"different but short", {}),
    ],
)
def test_invalid_download_leaves_no_partial_cache(tmp_path, manifest, content, headers):
    def respond(request):
        return httpx.Response(200, content=content, headers=headers)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError):
            media.ensure_asset("test", tmp_path, download=True, manifest=manifest, client=client)
    assert list(tmp_path.iterdir()) == []


def test_chunked_oversize_is_bounded_without_content_length(tmp_path, manifest):
    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            yield b"X" * 65536
            pytest.fail("Downloader read beyond the first oversized chunk")

    def respond(request):
        return httpx.Response(200, stream=Chunks())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="exceeds"):
            media.ensure_asset("test", tmp_path, download=True, manifest=manifest, client=client)
    assert list(tmp_path.iterdir()) == []


def test_tampered_cache_fails_without_replacing_it(tmp_path, manifest):
    path = tmp_path / "fixture.png"
    path.write_bytes(b"tampered test data!!")
    with pytest.raises(ValueError, match="hash/size"):
        media.ensure_asset("test", tmp_path, download=True, manifest=manifest)
    assert path.read_bytes() == b"tampered test data!!"


@pytest.mark.parametrize("filename", ["../escape.png", "a/b.png", "a\\b.png", "/a.png"])
def test_cache_traversal_is_rejected(tmp_path, manifest, filename):
    manifest["assets"]["test"]["filename"] = filename
    with pytest.raises(ValueError, match="basename"):
        media.ensure_asset("test", tmp_path, download=True, manifest=manifest)


def test_video_preparation_never_uses_label_window_to_crop(tmp_path, manifest):
    (tmp_path / "fixture.png").write_bytes(b"licensed test fixture")
    first = {"asset": "test", "kind": "video", "start_seconds": 13, "end_seconds": 17}
    second = {"asset": "test", "kind": "video", "start_seconds": 19, "end_seconds": 23}
    assert media.prepare_case(first, tmp_path, manifest=manifest) == media.prepare_case(
        second, tmp_path, manifest=manifest
    )


def test_derived_cache_hash_and_source_provenance_are_checked(tmp_path, manifest):
    (tmp_path / "fixture.png").write_bytes(b"licensed test fixture")
    page = tmp_path / "table.png"
    page.write_bytes(b"rendered page")
    case = {"asset": "test", "render": {"page": 6, "dpi": 130, "filename": "table.png"}}
    provenance = {
        "source_sha256": manifest["assets"]["test"]["sha256"],
        "page": 6,
        "dpi": 130,
        "sha256": media.sha256_file(page),
        "renderer": "test-renderer",
    }
    (tmp_path / "table.png.provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    assert media.prepare_case(case, tmp_path, manifest=manifest) == page
    page.write_bytes(b"modified page")
    with pytest.raises(ValueError, match="Rendered media hash"):
        media.prepare_case(case, tmp_path, manifest=manifest)


def test_frozen_manifest_contains_provenance_and_nonoverlapping_video_cases():
    data = media.load_manifest()
    cases = data["cases"]
    assert [
        sum(case["kind"] == kind for case in cases) for kind in ("photo", "table", "video")
    ] == [2, 2, 2]
    for asset in data["assets"].values():
        media._validate_source(asset)
        assert asset["author"] and asset["license"] and asset["license_url"] and asset["revision"]
    videos = [case for case in cases if case["kind"] == "video"]
    assert videos[0]["end_seconds"] < videos[1]["start_seconds"]
    assert videos[0]["asset"] == videos[1]["asset"]
    assert data["assets"][videos[0]["asset"]]["media_type"] == "animation"
    for case in cases:
        assert case["question"] and case["reference_answer"] and case["expected_any_groups"]
