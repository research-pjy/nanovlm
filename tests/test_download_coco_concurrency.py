"""download_selected used to be a single urlretrieve-per-image loop, which
took ~1.5 days for ~9,500/28,100 images in practice (network round-trip
latency, not bandwidth, was the bottleneck). Now it fans out across a
thread pool — these tests mock urlretrieve (no real network) and check
the concurrent path is still correct: right counts, no torn/half-written
files from workers racing on the same temp path, skip-existing still works.
"""

import threading
import time
from pathlib import Path
from unittest.mock import patch

from scripts.download_coco import download_selected


def _fake_pool(n=60):
    return {
        i: {"file_name": f"{i:012d}.jpg", "split": "train2017", "coco_captions": ["x"] * 5}
        for i in range(n)
    }


def _fake_selection(pool):
    ids = sorted(pool.keys())
    return {"held_out": ids[:10], "train": ids[10:50], "val": ids[50:]}


def test_concurrent_download_writes_all_files(tmp_path):
    pool = _fake_pool()
    selection = _fake_selection(pool)
    images_dir = tmp_path / "images"

    call_count = {"n": 0}
    lock = threading.Lock()

    def fake_urlretrieve(url, dest):
        with lock:
            call_count["n"] += 1
        time.sleep(0.005)  # simulate network latency, exercises real concurrency
        Path(dest).write_bytes(b"fake-jpeg-bytes")

    with patch("scripts.download_coco.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        downloaded, skipped, failed = download_selected(pool, selection, images_dir, workers=8)

    assert downloaded == 60
    assert skipped == 0
    assert failed == 0
    assert call_count["n"] == 60

    for image_id, meta in pool.items():
        dest = images_dir / meta["split"] / meta["file_name"]
        assert dest.exists()
        assert dest.read_bytes() == b"fake-jpeg-bytes"
        # no leftover .part temp files
        assert not any(images_dir.rglob(f"*{image_id}.part"))


def test_concurrent_download_skips_existing(tmp_path):
    pool = _fake_pool(n=20)
    selection = _fake_selection({k: v for k, v in pool.items() if k < 20})
    selection = {"held_out": [], "train": list(pool.keys())[:15], "val": list(pool.keys())[15:]}
    images_dir = tmp_path / "images"

    # pre-populate half the files as if a prior run already got them
    preexisting = list(pool.keys())[:10]
    for image_id in preexisting:
        meta = pool[image_id]
        d = images_dir / meta["split"]
        d.mkdir(parents=True, exist_ok=True)
        (d / meta["file_name"]).write_bytes(b"already-here")

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(b"newly-downloaded")

    with patch("scripts.download_coco.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        downloaded, skipped, failed = download_selected(pool, selection, images_dir, workers=4)

    assert skipped == 10
    assert downloaded == 10
    assert failed == 0
    # pre-existing files were left untouched, not re-downloaded
    for image_id in preexisting:
        meta = pool[image_id]
        assert (images_dir / meta["split"] / meta["file_name"]).read_bytes() == b"already-here"


def test_concurrent_download_reports_failures_without_crashing(tmp_path):
    pool = _fake_pool(n=10)
    selection = {"held_out": [], "train": list(pool.keys()), "val": []}
    images_dir = tmp_path / "images"

    def flaky_urlretrieve(url, dest):
        if "000000000003" in url:
            raise OSError("simulated network failure")
        Path(dest).write_bytes(b"ok")

    with patch("scripts.download_coco.urllib.request.urlretrieve", side_effect=flaky_urlretrieve), \
         patch("scripts.download_coco.time.sleep"):  # skip real backoff delay in the test
        downloaded, skipped, failed = download_selected(pool, selection, images_dir, workers=4)

    assert failed == 1
    assert downloaded == 9
    assert skipped == 0
