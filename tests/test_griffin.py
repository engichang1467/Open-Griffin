"""Checks for the parts of Griffin that do not need a GPU or model weights."""

import torch
import torch.nn.functional as F

from griffin.attn import (
    GriffinAttnProcessor,
    GriffinState,
    SourceCache,
    alpha_schedule,
    flatten_mask,
    shared_attention,
)
from griffin.invert import find_token, subject_mask
from griffin.layout import (
    box_mask,
    correspond,
    disjoin,
    farthest_points,
    mix_features,
    save_mask,
    select_keypoints,
    update_layouts,
)
from griffin.pipeline import _ip_masks, ip_scale
from run import build_layout, depth_sorted


def test_flatten_mask():
    layout = torch.zeros(8, 8, dtype=torch.bool)
    layout[:4] = True  # top half
    assert flatten_mask(layout, 64).tolist() == list(range(32))
    # a coarser attention layer gets the same region, nearest-downsampled
    assert flatten_mask(layout, 16).tolist() == list(range(8))


def test_shared_attention():
    torch.manual_seed(0)
    b, heads, side, d = 2, 4, 8, 16
    seq = side * side
    q, k, v = (torch.randn(b, heads, seq, d) for _ in range(3))

    layout = torch.zeros(side, side, dtype=torch.bool)
    layout[:4] = True
    idx = flatten_mask(layout, seq)
    bg = torch.arange(seq // 2, seq)  # everything the layout leaves out
    k_src, v_src = torch.randn(1, heads, 9, d), torch.randn(1, heads, 9, d)
    alpha = 0.7

    out = shared_attention(q, k, v, [(idx, k_src, v_src)], bg, alpha)
    plain = F.scaled_dot_product_attention(q, k, v)

    # background rows keep plain target self-attention (eq. 8)
    assert torch.allclose(out[:, :, bg], plain[:, :, bg], atol=1e-5)

    # region rows attend over [alpha * source] ++ [own target rows ++ background]
    ctx = torch.cat([idx, bg])
    k_hat = torch.cat([alpha * k_src.expand(b, -1, -1, -1), k[:, :, ctx]], dim=2)
    v_hat = torch.cat([v_src.expand(b, -1, -1, -1), v[:, :, ctx]], dim=2)
    want = F.scaled_dot_product_attention(q[:, :, idx], k_hat, v_hat)
    assert torch.allclose(out[:, :, idx], want, atol=1e-5)
    assert not torch.allclose(out[:, :, idx], plain[:, :, idx], atol=1e-3)

    # alpha only scales the source logits, so alpha=0 still mixes in V_S
    zero = shared_attention(q, k, v, [(idx, k_src, v_src)], bg, 0.0)
    assert not torch.allclose(zero[:, :, idx], plain[:, :, idx], atol=1e-3)


def test_shared_attention_isolates_regions():
    """Region 0 must not change when region 1's source changes."""
    torch.manual_seed(1)
    b, heads, side, d = 1, 2, 8, 16
    seq = side * side
    q, k, v = (torch.randn(b, heads, seq, d) for _ in range(3))

    top, bottom = torch.zeros(side, side, dtype=torch.bool), torch.zeros(side, side, dtype=torch.bool)
    top[:4], bottom[4:] = True, True
    i0, i1 = flatten_mask(top, seq), flatten_mask(bottom, seq)
    none = torch.empty(0, dtype=torch.long)  # the two layouts tile the image
    src = [torch.randn(1, heads, 6, d) for _ in range(4)]

    a = shared_attention(q, k, v, [(i0, src[0], src[1]), (i1, src[2], src[3])], none, 0.9)
    b_ = shared_attention(q, k, v, [(i0, src[0], src[1]), (i1, torch.randn(1, heads, 6, d), src[3])], none, 0.9)
    assert torch.allclose(a[:, :, i0], b_[:, :, i0], atol=1e-6)
    assert not torch.allclose(a[:, :, i1], b_[:, :, i1], atol=1e-3)


def test_region_attends_to_background_not_other_regions():
    """Eq. 6/7: K^n_T covers layout component n and the background, nothing else.

    So background content has to reach a region's rows, while a neighbouring
    region's content must not. Both halves matter -- widening the gather to the
    whole target would satisfy the first and break the second.
    """
    torch.manual_seed(2)
    b, heads, side, d = 1, 2, 8, 16
    seq = side * side
    q, k, v = (torch.randn(b, heads, seq, d) for _ in range(3))

    top, middle = torch.zeros(side, side, dtype=torch.bool), torch.zeros(side, side, dtype=torch.bool)
    top[:2], middle[2:4] = True, True  # rows 4-7 are left over as background
    i0, i1 = flatten_mask(top, seq), flatten_mask(middle, seq)
    bg = torch.arange(seq // 2, seq)
    src = [torch.randn(1, heads, 6, d) for _ in range(4)]
    regions = [(i0, src[0], src[1]), (i1, src[2], src[3])]

    base = shared_attention(q, k, v, regions, bg, 0.9)

    moved_bg = k.clone()
    moved_bg[:, :, bg] = torch.randn(b, heads, bg.numel(), d)
    assert not torch.allclose(base[:, :, i0], shared_attention(q, moved_bg, v, regions, bg, 0.9)[:, :, i0], atol=1e-3)

    moved_neighbour = k.clone()
    moved_neighbour[:, :, i1] = torch.randn(b, heads, i1.numel(), d)
    assert torch.allclose(base[:, :, i0], shared_attention(q, moved_neighbour, v, regions, bg, 0.9)[:, :, i0], atol=1e-6)


def test_shared_attention_writes_only_its_region():
    """The widened gather must not widen the write: rows outside `idx` stay plain."""
    torch.manual_seed(3)
    b, heads, side, d = 1, 2, 8, 16
    seq = side * side
    q, k, v = (torch.randn(b, heads, seq, d) for _ in range(3))

    layout = torch.zeros(side, side, dtype=torch.bool)
    layout[:2] = True
    idx = flatten_mask(layout, seq)
    bg = torch.arange(idx.numel(), seq)

    out = shared_attention(q, k, v, [(idx, torch.randn(1, heads, 6, d), torch.randn(1, heads, 6, d))], bg, 0.9)
    plain = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out[:, :, bg], plain[:, :, bg], atol=1e-5)
    assert not torch.allclose(out[:, :, idx], plain[:, :, idx], atol=1e-3)


def test_shared_attention_empty_background():
    """An empty `bg` is a plain concat with an empty index, not a special case."""
    torch.manual_seed(4)
    b, heads, side, d = 1, 2, 8, 16
    seq = side * side
    q, k, v = (torch.randn(b, heads, seq, d) for _ in range(3))

    layout = torch.ones(side, side, dtype=torch.bool)
    idx = flatten_mask(layout, seq)
    k_src, v_src = torch.randn(1, heads, 6, d), torch.randn(1, heads, 6, d)

    out = shared_attention(q, k, v, [(idx, k_src, v_src)], torch.empty(0, dtype=torch.long), 0.9)
    k_hat = torch.cat([0.9 * k_src.expand(b, -1, -1, -1), k], dim=2)
    v_hat = torch.cat([v_src.expand(b, -1, -1, -1), v], dim=2)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k_hat, v_hat), atol=1e-5)


def test_regions_background_is_the_complement():
    """`_regions` hands back exactly the pixels no layout claimed, and no others."""
    heads, side, d = 2, 4, 8
    seq = side * side
    layouts = [torch.zeros(8, 8, dtype=torch.bool) for _ in range(2)]
    layouts[0][:2], layouts[1][2:4] = True, True  # bottom half unclaimed

    caches = []
    for _ in layouts:
        cache = SourceCache()
        cache.put("attn1", 0, torch.randn(1, heads, seq, d), torch.randn(1, heads, seq, d))
        caches.append(cache)

    state = GriffinState(mode="share", t=0, sources=caches, layouts=layouts)
    regions, bg = GriffinAttnProcessor(state, "attn1")._regions(seq, torch.device("cpu"), torch.float32)

    claimed = torch.cat([idx for idx, _, _ in regions])
    assert sorted(claimed.tolist() + bg.tolist()) == list(range(seq)), "every pixel is claimed or background"
    assert set(claimed.tolist()).isdisjoint(bg.tolist()), "a key must not sit in a region and the background"


def test_alpha_schedule():
    assert alpha_schedule(5) == 0.0  # nothing shared before T_LBA
    assert alpha_schedule(10) > alpha_schedule(49)  # decays as t -> 0
    assert abs(alpha_schedule(10) - 1.2) < 0.01
    assert abs(alpha_schedule(50) - 0.4) < 1e-6


def test_find_token():
    assert find_token([49406, 320, 1929, 49407], [1929]) == 2
    assert find_token([49406, 320, 1929, 2368, 49407], [1929, 2368]) == 2
    try:
        find_token([1, 2], [9])
    except ValueError:
        pass
    else:
        raise AssertionError("missing subject should raise")


def test_subject_mask():
    cache = SourceCache()
    blob = torch.full((16, 16), 0.1)
    blob[:8, :8] = 1.0  # bright quadrant
    cache.cross["up_blocks.1.attn2"] = blob.flatten()

    mask = subject_mask(cache, size=32)
    assert mask.shape == (32, 32)
    assert mask[:12, :12].all(), "blob interior should be inside the mask"
    assert not mask[20:, 20:].any(), "far background should be outside the mask"


def test_ip_scale():
    assert ip_scale(0) == 1.8  # structure init
    assert ip_scale(9) == 1.8
    assert ip_scale(10) == 0.8  # T_LBA
    assert ip_scale(29) == 0.8
    assert ip_scale(30) == 0.4
    assert ip_scale(49) == 0.4


def test_ip_masks_shape():
    """diffusers wants one [1, num_images, H, W] tensor per loaded adapter."""
    layouts = [torch.zeros(16, 16, dtype=torch.bool) for _ in range(3)]
    layouts[0][:8] = True
    masks = _ip_masks(layouts, device="cpu", dtype=torch.float32)
    assert len(masks) == len(layouts), "list length must equal the adapter count"
    for mask in masks:
        assert mask.ndim == 4 and mask.shape[:2] == (1, 1)
        assert mask.dtype == torch.float32
    assert masks[0].sum() == 8 * 16


def test_mix_features():
    dift = torch.randn(12, 8) * 100  # wildly different magnitudes
    dino = torch.randn(12, 5) * 0.01
    mixed = mix_features(dift, dino, beta=0.5)
    assert mixed.shape == (12, 13)
    assert torch.allclose(mixed.norm(dim=-1), torch.ones(12), atol=1e-5)
    # beta must actually shift the balance, not just rescale
    dift_heavy = mix_features(dift, dino, beta=0.9)
    assert dift_heavy[:, :8].abs().sum() > mixed[:, :8].abs().sum()


def test_correspond():
    a, b = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    score, source_id = correspond(target, [a, b])
    assert source_id.tolist() == [0, 1, 1]
    assert torch.allclose(score, torch.tensor([1.0, 1.0, 0.8]), atol=1e-6)


def test_farthest_points():
    coords = torch.tensor([[0, 0], [0, 1], [7, 7], [7, 0], [0, 7]])
    picked = farthest_points(coords, 3).tolist()
    assert picked[0] == [0, 0]  # seeds on the first (highest-scoring) point
    assert [0, 1] not in picked  # the near-duplicate loses to the corners
    assert len(picked) == 3
    # fewer candidates than requested: return them all
    assert farthest_points(coords[:2], 5).shape == (2, 2)


def test_select_keypoints():
    side = 8
    score = torch.full((side * side,), 0.05)
    score.view(side, side)[:3, :3] = 0.9  # subject 0, top-left
    score.view(side, side)[5:, 5:] = 0.9  # subject 1, bottom-right
    source_id = torch.zeros(side * side, dtype=torch.long)
    source_id.view(side, side)[4:, :] = 1

    groups = select_keypoints(score, source_id, n_sources=2, ratio=0.5, k=5)
    assert len(groups) == 2
    for coords, rows, cols in ((groups[0], range(3), range(3)), (groups[1], range(5, 8), range(5, 8))):
        assert 0 < coords.shape[0] <= 5
        for r, c in coords.tolist():
            assert r in rows and c in cols, "Otsu should have dropped the low-score background"

    # a subject with no confident match yields nothing rather than garbage
    empty = select_keypoints(score, torch.full_like(source_id, 0), n_sources=2)
    assert empty[1].shape == (0, 2)


def test_box_mask():
    mask = box_mask([0.0, 0.0, 0.5, 0.5], 8, 8)
    assert mask[:4, :4].all()
    assert mask.sum() == 16
    # a sliver still has to survive rounding, or the subject vanishes
    assert box_mask([0.0, 0.0, 0.01, 0.01], 8, 8).sum() == 1
    for bad in ([0.5, 0.0, 0.5, 1.0], [0.9, 0.0, 0.1, 1.0]):
        try:
            box_mask(bad, 8, 8)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should be rejected")


def test_build_layout():
    assert build_layout({"subject": "dog", "box": [0.0, 0.0, 1.0, 0.5]}, 8, 8).sum() == 32
    for bad in ({"subject": "dog"}, {"subject": "dog", "box": [0, 0, 1, 1], "mask": "m.png"}):
        try:
            build_layout(bad, 8, 8)
        except ValueError:
            pass
        else:
            raise AssertionError("need exactly one of box/mask")


def test_disjoin():
    """Front keeps everything, back loses exactly the intersection."""
    front = box_mask([0.0, 0.0, 0.6, 1.0], 10, 10)
    back = box_mask([0.4, 0.0, 1.0, 1.0], 10, 10)
    overlap = (front & back).sum()
    assert overlap > 0, "this test is pointless if the boxes do not actually overlap"

    a, b = disjoin([front, back])
    assert not (a & b).any(), "masks must be disjoint afterwards"
    assert torch.equal(a, front), "the frontmost mask keeps its full area"
    assert b.sum() == back.sum() - overlap
    assert torch.equal(a | b, front | back), "subtraction must not lose pixels to neither subject"


def test_disjoin_chain():
    """The third mask yields to both predecessors, not just the one before it."""
    masks = [box_mask(box, 12, 12) for box in ([0.0, 0.0, 0.5, 0.5], [0.25, 0.0, 0.75, 0.5], [0.0, 0.0, 1.0, 1.0])]
    a, b, c = disjoin(masks)
    assert not (a & b).any() and not (a & c).any() and not (b & c).any()
    assert not (c & masks[0]).any(), "the back mask must yield to the frontmost, not only its neighbour"


def test_disjoin_leaves_non_overlapping_masks_alone():
    """The common case is a no-op, so specs without overlap generate exactly as before."""
    masks = [box_mask([0.0, 0.0, 0.4, 1.0], 8, 8), box_mask([0.6, 0.0, 1.0, 1.0], 8, 8)]
    for before, after in zip(masks, disjoin(masks)):
        assert torch.equal(before, after)


def test_disjoin_can_empty_a_fully_occluded_mask():
    masks = [box_mask([0.0, 0.0, 1.0, 1.0], 8, 8), box_mask([0.2, 0.2, 0.4, 0.4], 8, 8)]
    front, behind = disjoin(masks)
    assert front.all()
    assert not behind.any(), "a subject entirely behind another has nothing left"


def test_depth_sorted():
    subjects = [{"subject": "dog"}, {"subject": "eagle"}, {"subject": "cat"}]
    assert [s["subject"] for s in depth_sorted(subjects)] == ["dog", "eagle", "cat"], "no order key is a no-op"

    ordered = [{"subject": "dog", "order": 2}, {"subject": "eagle", "order": 0}, {"subject": "cat", "order": 1}]
    assert [s["subject"] for s in depth_sorted(ordered)] == ["eagle", "cat", "dog"], "order 0 is frontmost"

    # half-declared depth is ambiguous -- an explicit order 0 and a default
    # position 0 are the same number -- so it is rejected instead of guessed
    try:
        depth_sorted([{"subject": "dog"}, {"subject": "eagle", "order": 0}])
    except ValueError as error:
        assert "dog" in str(error), "the error should name the subject that is missing an order"
    else:
        raise AssertionError("a partially ordered spec should be rejected")


def test_update_layouts_returns_disjoint_masks():
    """SAM re-derives each mask alone, so `update_layouts` has to re-partition them."""

    class Preview:
        size = (16, 16)

    overlapping = [box_mask([0.0, 0.0, 0.7, 1.0], 16, 16), box_mask([0.3, 0.0, 1.0, 1.0], 16, 16)]
    calls = iter(overlapping)

    # a fresh feature map per call: identical ones would make every correspondence
    # score 1.0, and Otsu cannot threshold a range of zero
    generator = torch.Generator().manual_seed(0)

    def extractor(latents, embeds, beta=0.5):
        return F.normalize(torch.randn(1, 64, 8, generator=generator), dim=-1)

    layouts = [box_mask([0.0, 0.0, 0.5, 1.0], 16, 16), box_mask([0.5, 0.0, 1.0, 1.0], 16, 16)]
    out = update_layouts(
        extractor,
        lambda image, points: next(calls),
        torch.randn(1, 4, 8, 8),
        Preview(),
        [torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)],
        [torch.ones(8, 8, dtype=torch.bool), torch.ones(8, 8, dtype=torch.bool)],
        layouts,
        torch.randn(1, 77, 8),
    )
    assert len(out) == 2
    assert not (out[0] & out[1]).any(), "refined masks must not overlap"
    assert torch.equal(out[0], overlapping[0]), "the frontmost refined mask is kept whole"


def test_save_mask():
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image

    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:4] = True  # top half

    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "plain.png"
        save_mask(mask, plain)
        assert Image.open(plain).size == (8, 8)

        # the overlay keeps the image's own size, upsampling the mask to reach it
        over = Path(tmp) / "over.png"
        base = Image.new("RGB", (32, 32), (0, 0, 255))
        save_mask(mask, over, base)
        pixels = np.asarray(Image.open(over))
        assert pixels.shape == (32, 32, 3)
        assert pixels[:16].mean(axis=(0, 1))[0] > 100, "top half should be tinted red"
        assert pixels[16:].mean(axis=(0, 1))[0] < 10, "bottom half should be untouched"


def test_example_spec_parses():
    import json
    from pathlib import Path

    spec = json.loads((Path(__file__).parent.parent / "examples/dog_eagle.json").read_text())
    assert spec["prompt"] and spec["subjects"]
    for subject in spec["subjects"]:
        assert subject["subject"] in subject["prompt"], "subject word must appear in its source prompt"
        assert build_layout(subject, 64, 64).any()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
