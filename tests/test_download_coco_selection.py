"""The deterministic image-selection logic (DESIGN_DECISIONS.md §6-7) is
the part that MUST be reproducible and identical across both architecture
runs — test it directly, without touching the network.
"""

from scripts.download_coco import make_selection


def _fake_pool(n=1000):
    return {i: {"file_name": f"{i:012d}.jpg", "split": "train2017", "coco_captions": ["x"] * 5} for i in range(n)}


def test_selection_is_deterministic_for_same_seed():
    pool = _fake_pool()
    sel1 = make_selection(pool, num_images=500, num_held_out=50, seed=42, train_fraction=0.9)
    sel2 = make_selection(pool, num_images=500, num_held_out=50, seed=42, train_fraction=0.9)
    assert sel1 == sel2


def test_different_seed_gives_different_selection():
    pool = _fake_pool()
    sel1 = make_selection(pool, num_images=500, num_held_out=50, seed=42, train_fraction=0.9)
    sel2 = make_selection(pool, num_images=500, num_held_out=50, seed=1, train_fraction=0.9)
    assert sel1 != sel2


def test_held_out_train_val_are_disjoint_and_correctly_sized():
    pool = _fake_pool()
    sel = make_selection(pool, num_images=500, num_held_out=50, seed=42, train_fraction=0.9)

    held_out, train, val = set(sel["held_out"]), set(sel["train"]), set(sel["val"])
    assert len(sel["held_out"]) == 50
    assert len(sel["train"]) + len(sel["val"]) == 500
    assert len(sel["train"]) == 450
    assert len(sel["val"]) == 50

    assert held_out.isdisjoint(train)
    assert held_out.isdisjoint(val)
    assert train.isdisjoint(val)


def test_held_out_selected_before_train_val_pool():
    """Held-out must be carved out BEFORE the num_images pool is taken, so
    shrinking/growing num_images never changes which images are held out
    (DESIGN_DECISIONS.md §6: 'held out first, pre-limit selection').
    """
    pool = _fake_pool()
    sel_small = make_selection(pool, num_images=100, num_held_out=50, seed=42, train_fraction=0.9)
    sel_large = make_selection(pool, num_images=500, num_held_out=50, seed=42, train_fraction=0.9)
    assert sel_small["held_out"] == sel_large["held_out"]


def test_raises_when_pool_too_small():
    pool = _fake_pool(n=100)
    try:
        make_selection(pool, num_images=90, num_held_out=50, seed=42, train_fraction=0.9)
        assert False, "expected ValueError for oversubscribed pool"
    except ValueError:
        pass
