"""Disagreement-head PLUMBING tests: checkpoint round-trip, DIS1 export, and
the warm-start migrations.

The head itself (two dense layers) is not what breaks. What breaks is the
carrying: a tensor that is trained and then silently not saved, an arch change
that makes a warm start refuse, or a re-init that leaves the head dead. Each of
those is a whole training run wasted, and none of them shows up in a loss curve
— the head is `stop_gradient`ed away from the trunk on both sides, so a broken
one costs nothing observable until someone reads its output.

Run:  cd jax && JAX_PLATFORMS=cpu uv run --no-project --with numpy --with jax \
        --with optax --with pytest -- python -m pytest tests/test_dis_head.py -q
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax import checkpoint as ck  # noqa: E402
from archon_jax.spec import ArchSpec, TrainConfig  # noqa: E402


def dis_arch(dis_proj=3, **kw):
    """A tiny but STRUCTURALLY REAL arch: every tensor name the full one has,
    trivial sizes. `psqt`/`out_skip` on, so DIS1 has to survive being the third
    trailer rather than the only one."""
    return dataclasses.replace(
        ArchSpec.full(), hl=16, l2=4, l3=4, num_buckets=2, policy=False,
        factorize=False, psqt=True, out_skip=True, dis_proj=dis_proj, **kw)


def fake_batch(arch, n=8, seed=0):
    from archon_jax.spec import HALFKA_DIMS, THREAT_DIMS
    rng = np.random.default_rng(seed)
    return {
        "hk_stm": rng.integers(0, HALFKA_DIMS, (n, 32), dtype=np.int32),
        "hk_ntm": rng.integers(0, HALFKA_DIMS, (n, 32), dtype=np.int32),
        "th_stm": rng.integers(0, THREAT_DIMS, (n, 32), dtype=np.int32),
        "th_ntm": rng.integers(0, THREAT_DIMS, (n, 32), dtype=np.int32),
        "bucket": rng.integers(0, arch.num_buckets, (n,), dtype=np.int32),
        "target": rng.random((n,), dtype=np.float32),
    }


def small_params(arch, seed=7):
    """Random params in a range every export encoding can hold."""
    rng = np.random.default_rng(seed)
    return {k: (0.1 * rng.standard_normal(s)).astype(np.float32)
            for k, s in arch.param_shapes().items()}


# ---------------------------------------------------------------------------
# 1. The head survives a checkpoint round-trip.
# ---------------------------------------------------------------------------

def test_raw_round_trip_is_bit_identical_after_training(tmp_path):
    """Train a few steps with dis_lambda > 0, save, load, compare BITS.

    `write_raw` is name-generic, so this would pass even if nobody had thought
    about the head — which is exactly why it is worth pinning: the next person
    to add a filtered save path has an oracle."""
    from archon_jax.train import make_update_step
    import jax.numpy as jnp

    arch = dis_arch()
    cfg = dataclasses.replace(TrainConfig(), dis_lambda=0.5, pol_lambda=0.0,
                              batches_per_superbatch=4, superbatches=1)
    opt, step = make_update_step(arch, cfg)
    params = {k: jnp.asarray(v) for k, v in small_params(arch).items()}
    state = opt.init(params)
    for i in range(3):
        params, state, metrics = step(params, state, fake_batch(arch, seed=i))
    # The head really did train: its loss and both nulls are reported.
    for k in ("dis", "dis_sgn", "dis_sgn_null", "dis_mag", "dis_mag_null"):
        assert k in metrics, f"{k} missing from the training metrics"

    host = {k: np.asarray(v) for k, v in params.items()}
    path = tmp_path / "ck-raw.bin"
    ck.write_raw(path, host)
    back = ck.open_raw(path, shapes=arch.param_shapes(), mmap=False)

    assert set(back) == set(arch.param_shapes())
    for name in ck.DIS_TENSORS:
        assert name in back, f"{name} did not survive the round-trip"
        assert np.array_equal(back[name], host[name]), f"{name} changed"
        # A head that trained to exactly its init would make the assert above
        # vacuous; assert it MOVED.
        assert not np.array_equal(back[name], small_params(arch)[name]), (
            f"{name} is unchanged by training — the dis loss is not reaching it")
    for name, arr in host.items():
        assert np.array_equal(back[name], arr), f"{name} changed"


def test_dis_tensor_shapes_are_the_two_column_contract():
    arch = dis_arch(dis_proj=5)
    s = arch.param_shapes()
    assert s["dis_projw"] == (arch.hl, 5)
    assert s["dis_projb"] == (5,)
    assert s["dis_outw"] == (5, ck.DIS_OUTS)
    assert s["dis_outb"] == (ck.DIS_OUTS,)


# ---------------------------------------------------------------------------
# 2. The head survives export (the DIS1 section).
# ---------------------------------------------------------------------------

def test_quant_export_carries_dis1_verbatim(tmp_path):
    arch = dis_arch(dis_proj=3)
    params = small_params(arch)
    path = tmp_path / "quant.bin"
    ck.write_quant(path, params, arch)
    raw = path.read_bytes()

    i = raw.find(ck.DIS_MAGIC)
    assert i > 0, "no DIS1 section in the quant stream"
    # It rides LAST: after the padded stream, PSQ1 and SKP1.
    assert raw.find(ck.PSQT_MAGIC) < raw.find(ck.SKIP_MAGIC) < i
    assert i + ck.dis_section_len(raw[i:]) == len(raw), "DIS1 is not the final section"

    got = ck.parse_dis_section(raw[i:])
    for name in ck.DIS_TENSORS:
        # f32 on disk and f32 in memory: exports EXACTLY, no tolerance.
        assert np.array_equal(got[name], params[name]), f"{name} changed on export"


def test_a_dis_less_arch_exports_byte_identically(tmp_path):
    """The compatibility claim, asserted: turning the head off reproduces the
    old bytes exactly, so an older net has nothing new to parse."""
    arch = dis_arch(dis_proj=3)
    plain = dataclasses.replace(arch, dis_proj=0)
    params = small_params(arch)
    a, b = tmp_path / "with.bin", tmp_path / "without.bin"
    ck.write_quant(a, params, arch)
    ck.write_quant(b, {k: v for k, v in params.items()
                       if k not in ck.DIS_TENSORS}, plain)
    with_b, without_b = a.read_bytes(), b.read_bytes()
    assert with_b[:len(without_b)] == without_b
    assert ck.DIS_MAGIC not in without_b


def test_dequantise_accepts_the_dis1_trailer(tmp_path):
    """`dequantise_quant` parses the STREAM only, but it validates the trailer;
    an unrecognised one is a hard error, so DIS1 has to be whitelisted."""
    arch = dis_arch(dis_proj=3)
    path = tmp_path / "quant.bin"
    ck.write_quant(path, small_params(arch), arch)
    out = ck.dequantise_quant(path, arch)
    assert "l1w" in out and not (set(out) & set(ck.DIS_TENSORS))


def test_netpack_trailer_split_accepts_dis1(tmp_path):
    """The .cenn packer's trailer validator must pass a PSQ1+SKP1+DIS1 tail
    through verbatim — and must still REJECT a corrupt one."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/nnue"))
    import netpack

    arch = dis_arch(dis_proj=3)
    path = tmp_path / "quant.bin"
    ck.write_quant(path, small_params(arch), arch)
    raw = path.read_bytes()

    sizes = {"i16": 2, "i8": 1, "f32": 4}
    shapes = arch.param_shapes()
    end = sum(int(np.prod(shapes[n])) * sizes[k]
              for n, k, _, _ in arch.quant_layout())

    sec = netpack._split_psq1_trailer(raw, end)
    assert sec is not None, "a PSQ1+SKP1+DIS1 trailer was rejected"
    assert sec.startswith(ck.PSQT_MAGIC)
    assert ck.DIS_MAGIC in sec
    # Truncating the DIS1 payload must be REJECTED, not silently accepted.
    assert netpack._split_psq1_trailer(raw[:-4], end) is None
    # And so must an unknown section appended after it.
    assert netpack._split_psq1_trailer(raw + b"XXXX", end) is None


# ---------------------------------------------------------------------------
# 3. Warm start: the migrations that decide whether a run can even launch.
# ---------------------------------------------------------------------------

def test_exact_warm_start_is_bit_identical(tmp_path):
    arch = dis_arch(dis_proj=3)
    params = small_params(arch)
    path = tmp_path / "raw.bin"
    ck.write_raw(path, params)
    back = ck.load_warm(path, arch)
    for name in ck.DIS_TENSORS:
        assert np.array_equal(back[name], params[name]), f"{name} not resumed exactly"


def test_warm_start_reinitialises_a_dis_head_of_a_DIFFERENT_WIDTH(tmp_path):
    """**The one that blocks the launch.** `engines/canonical/`'s own raw
    checkpoint carries `dis_proj = 64`; a probe run at 32 must not be refused."""
    src = dis_arch(dis_proj=8)
    params = small_params(src)
    path = tmp_path / "raw.bin"
    ck.write_raw(path, params)

    dst = dis_arch(dis_proj=4)
    back = ck.load_warm(path, dst)
    for name, shape in dst.param_shapes().items():
        assert back[name].shape == shape, name
    # The value trunk came through untouched; only the head was redrawn.
    assert np.array_equal(back["l1w"], params["l1w"])
    # NOT ZERO: a zero projection sits on the CReLU kink and emits a constant.
    assert np.abs(back["dis_projw"]).max() > 0.0
    assert np.abs(back["dis_outw"]).max() > 0.0


def test_warm_start_adds_a_head_a_checkpoint_never_had(tmp_path):
    src = dis_arch(dis_proj=0)
    params = {k: v for k, v in small_params(src).items()}
    path = tmp_path / "raw.bin"
    ck.write_raw(path, params)

    dst = dis_arch(dis_proj=4)
    back = ck.load_warm(path, dst)
    assert back["dis_projw"].shape == (dst.hl, 4)
    assert np.abs(back["dis_projw"]).max() > 0.0
    assert np.array_equal(back["dis_projb"], np.zeros(4, np.float32))


def test_warm_start_widens_the_old_one_column_head_into_column_1(tmp_path):
    """The d7582485 shape change: `dis_outw (proj,)` -> `(proj, 2)`. The old
    lane predicted |residual| = TODAY'S COLUMN 1, so it must land there."""
    arch = dis_arch(dis_proj=4)
    params = small_params(arch)
    old = dict(params)
    old["dis_outw"] = np.arange(4, dtype=np.float32) + 1.0     # (4,)
    old["dis_outb"] = np.array([0.75], dtype=np.float32)       # (1,)
    path = tmp_path / "raw.bin"
    ck.write_raw(path, old)

    back = ck.load_warm(path, arch)
    assert back["dis_outw"].shape == (4, ck.DIS_OUTS)
    assert np.array_equal(back["dis_outw"][:, 1], old["dis_outw"])
    assert np.abs(back["dis_outw"][:, 0]).max() > 0.0          # fresh, not zero
    assert back["dis_outb"].tolist() == [0.0, 0.75]


def test_a_different_head_with_a_COLLIDING_ELEMENT_COUNT_is_not_reused(tmp_path):
    """**The bug this file exists for.** The canonical checkpoint's head is the
    pre-d7582485 shape at `dis_proj = 64`, so `dis_outw` holds 64 values — and
    an arch asking for `dis_proj = 32` wants `(32, 2)`, which is ALSO 64. A
    per-tensor size check accepts that and reshapes one net's 64-unit magnitude
    lane into another net's 32x2 output matrix, silently. `dis_projw` is what
    disambiguates: its count is `hl · proj`."""
    arch = dis_arch(dis_proj=8)
    old = small_params(arch)
    old["dis_projw"] = np.full((arch.hl, 8), 3.0, np.float32)
    old["dis_outw"] = np.arange(8, dtype=np.float32)      # old single lane
    old["dis_outb"] = np.array([0.5], np.float32)
    path = tmp_path / "raw.bin"
    ck.write_raw(path, old)

    # 4 * DIS_OUTS == 8 == the checkpoint's dis_outw count. The collision.
    dst = dis_arch(dis_proj=4)
    assert 4 * ck.DIS_OUTS == old["dis_outw"].size
    back = ck.load_warm(path, dst)
    assert back["dis_outw"].shape == (4, ck.DIS_OUTS)
    assert not np.array_equal(back["dis_outw"].ravel(), old["dis_outw"]), (
        "a 64-unit single-lane head was reshaped into a 32x2 output matrix")
    assert back["dis_outb"].tolist() != [0.0, 0.5]
    # All four are fresh, and none is dead.
    assert np.abs(back["dis_projw"]).max() > 0.0
    assert np.abs(back["dis_outw"]).max() > 0.0


def test_warm_start_drops_a_head_the_arch_does_not_want(tmp_path):
    arch = dis_arch(dis_proj=4)
    path = tmp_path / "raw.bin"
    ck.write_raw(path, small_params(arch))
    back = ck.load_warm(path, dataclasses.replace(arch, dis_proj=0))
    assert not (set(back) & set(ck.DIS_TENSORS))


def test_warm_start_still_refuses_a_genuinely_wrong_arch(tmp_path):
    """The relaxations above are scoped to the dis head. A VALUE tensor whose
    shape moved is still a hard error — that is a wrong-arch load."""
    arch = dis_arch(dis_proj=4)
    path = tmp_path / "raw.bin"
    ck.write_raw(path, small_params(arch))
    with pytest.raises(ValueError):
        ck.load_warm(path, dataclasses.replace(arch, l2=8))
