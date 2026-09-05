#!/usr/bin/env python3
"""Generate the tiny .pth test fixtures for vc-convert.

Run offline (never in CI) with a stock Python 3 — no torch needed: the
torch checkpoint format is just a ZIP of one pickle plus raw tensor blobs,
and this script fabricates the torch globals the pickle references.

Outputs (next to this script):
  tiny_v2_f0.pth   — miniature but structurally complete RVC v2 F0 checkpoint
  tiny_v1.pth      — v1 checkpoint (conversion must reject it)
  tiny_v2_nof0.pth — v2 checkpoint with f0=0 (conversion must reject it)

Regenerate with:  python make_tiny_pth.py
"""

import io
import pickle
import struct
import sys
import types
import zipfile
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Fake torch module tree so pickle emits the same globals a real torch
# checkpoint contains (torch._utils._rebuild_tensor_v2, torch.FloatStorage…).
# ---------------------------------------------------------------------------

torch = types.ModuleType("torch")
torch_utils = types.ModuleType("torch._utils")


def _rebuild_tensor_v2(*args):  # never called; only pickled by reference
    raise NotImplementedError


_rebuild_tensor_v2.__module__ = "torch._utils"
_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
torch_utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def _make_storage_class(name):
    cls = type(name, (), {})
    cls.__module__ = "torch"
    setattr(torch, name, cls)
    return cls


FloatStorage = _make_storage_class("FloatStorage")
HalfStorage = _make_storage_class("HalfStorage")
LongStorage = _make_storage_class("LongStorage")

torch._utils = torch_utils
sys.modules["torch"] = torch
sys.modules["torch._utils"] = torch_utils


class FakeTensor:
    """Pickles exactly like a torch tensor (storage via persistent id)."""

    _next_key = 0

    def __init__(self, shape, values, dtype="float32"):
        numel = 1
        for d in shape:
            numel *= d
        assert len(values) == numel, f"{shape} vs {len(values)} values"
        self.shape = tuple(shape)
        self.values = values
        self.dtype = dtype
        self.key = str(FakeTensor._next_key)
        FakeTensor._next_key = FakeTensor._next_key + 1

    def storage_bytes(self):
        if self.dtype == "float32":
            return struct.pack(f"<{len(self.values)}f", *self.values)
        if self.dtype == "float16":
            return struct.pack(f"<{len(self.values)}e", *self.values)
        if self.dtype == "int64":
            return struct.pack(f"<{len(self.values)}q", *self.values)
        raise ValueError(self.dtype)

    def storage_class(self):
        return {
            "float32": FloatStorage,
            "float16": HalfStorage,
            "int64": LongStorage,
        }[self.dtype]

    def contiguous_stride(self):
        stride = []
        acc = 1
        for d in reversed(self.shape):
            stride.append(acc)
            acc *= d
        return tuple(reversed(stride))

    def __reduce_ex__(self, protocol):
        return (
            _rebuild_tensor_v2,
            (
                _StorageRef(self),
                0,
                self.shape,
                self.contiguous_stride(),
                False,
                OrderedDict(),
            ),
        )


class _StorageRef:
    def __init__(self, tensor):
        self.tensor = tensor


class _Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            t = obj.tensor
            numel = len(t.values)
            return ("storage", t.storage_class(), t.key, "cpu", numel)
        return None


# ---------------------------------------------------------------------------
# Deterministic pseudo-random weights (LCG; keeps fixtures byte-stable).
# ---------------------------------------------------------------------------

_seed = 0x12345678


def _rand():
    global _seed
    _seed = (_seed * 1103515245 + 12345) & 0x7FFFFFFF
    return (_seed / 0x7FFFFFFF) * 0.2 - 0.1  # small values in [-0.1, 0.1)


def t(shape, dtype="float32", zero=False, values=None):
    numel = 1
    for d in shape:
        numel *= d
    if values is None:
        values = [0.0] * numel if zero else [_rand() for _ in range(numel)]
    return FakeTensor(shape, values, dtype)


# ---------------------------------------------------------------------------
# Miniature RVC v2 config + full generator state dict.
#
# Dims are tiny but the structure (layer names, weight_norm parametrization,
# layer counts derived from the config) matches a real
# SynthesizerTrnMs768NSFsid state dict, so the graph builder exercises every
# code path.
# ---------------------------------------------------------------------------

INTER = 8  # inter_channels
HIDDEN = 8  # hidden_channels
FILTER = 16  # filter_channels
HEADS = 2
LAYERS = 1  # encoder layers (real RVC: 6)
KERNEL = 3
GIN = 4  # gin_channels (real RVC: 256)
UPS = [10, 10, 2, 2]  # 40k-style upsample rates → frame hop 400
UPS_K = [16, 16, 4, 4]
UPS_INIT = 16  # upsample_initial_channel
RES_K = [3, 7, 11]
RES_D = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
SPK = 4  # speakers (emb_g rows)
SR = 40000
FEAT = 768  # v2 ContentVec width
WINDOW = 10  # attention relative window
WN_LAYERS = 3  # WaveNet layers inside each coupling
FLOWS = [0, 2, 4, 6]  # ResidualCouplingLayer indices (odd ones are Flips)
HARMONICS = 0  # NSF harmonic_num (0 in RVC) → l_linear in_features = 1

CONFIG = [
    1025,  # spec_channels
    32,  # segment_size
    INTER,
    HIDDEN,
    FILTER,
    HEADS,
    LAYERS,
    KERNEL,
    0.0,  # p_dropout
    "1",  # resblock
    RES_K,
    RES_D,
    UPS,
    UPS_INIT,
    UPS_K,
    109,  # spk_embed_dim (overridden by emb_g rows at parse time)
    GIN,
    SR,
]


def wn(base, w):
    """weight_norm pair in the modern parametrizations layout.

    PyTorch weight_norm defaults to dim=0, so weight_g is always
    [dim0, 1, 1, ...] — including ConvTranspose1d, whose dim0 is in_channels.
    """
    g_shape = [w.shape[0]] + [1] * (len(w.shape) - 1)
    return {
        f"{base}.parametrizations.weight.original0": t(g_shape),
        f"{base}.parametrizations.weight.original1": w,
    }


def build_weights():
    w = {}

    # --- enc_p: TextEncoder768 -------------------------------------------
    w["enc_p.emb_phone.weight"] = t([HIDDEN, FEAT], dtype="float16")
    w["enc_p.emb_phone.bias"] = t([HIDDEN])
    w["enc_p.emb_pitch.weight"] = t([256, HIDDEN])
    head_dim = HIDDEN // HEADS
    for i in range(LAYERS):
        a = f"enc_p.encoder.attn_layers.{i}"
        for conv in ("conv_q", "conv_k", "conv_v", "conv_o"):
            w[f"{a}.{conv}.weight"] = t([HIDDEN, HIDDEN, 1])
            w[f"{a}.{conv}.bias"] = t([HIDDEN])
        w[f"{a}.emb_rel_k"] = t([1, 2 * WINDOW + 1, head_dim])
        w[f"{a}.emb_rel_v"] = t([1, 2 * WINDOW + 1, head_dim])
        w[f"enc_p.encoder.norm_layers_1.{i}.gamma"] = t([HIDDEN])
        w[f"enc_p.encoder.norm_layers_1.{i}.beta"] = t([HIDDEN])
        f = f"enc_p.encoder.ffn_layers.{i}"
        w[f"{f}.conv_1.weight"] = t([FILTER, HIDDEN, KERNEL])
        w[f"{f}.conv_1.bias"] = t([FILTER])
        w[f"{f}.conv_2.weight"] = t([HIDDEN, FILTER, KERNEL])
        w[f"{f}.conv_2.bias"] = t([HIDDEN])
        w[f"enc_p.encoder.norm_layers_2.{i}.gamma"] = t([HIDDEN])
        w[f"enc_p.encoder.norm_layers_2.{i}.beta"] = t([HIDDEN])
    w["enc_p.proj.weight"] = t([INTER * 2, HIDDEN, 1])
    w["enc_p.proj.bias"] = t([INTER * 2])

    # --- flow: ResidualCouplingBlock (4 coupling layers, WN inside) ------
    half = INTER // 2
    for i in FLOWS:
        base = f"flow.flows.{i}"
        w[f"{base}.pre.weight"] = t([HIDDEN, half, 1])
        w[f"{base}.pre.bias"] = t([HIDDEN])
        for l in range(WN_LAYERS):
            w.update(wn(f"{base}.enc.in_layers.{l}", t([2 * HIDDEN, HIDDEN, 5])))
            out_ch = 2 * HIDDEN if l < WN_LAYERS - 1 else HIDDEN
            w[f"{base}.enc.in_layers.{l}.bias"] = t([2 * HIDDEN])
            w.update(wn(f"{base}.enc.res_skip_layers.{l}", t([out_ch, HIDDEN, 1])))
            w[f"{base}.enc.res_skip_layers.{l}.bias"] = t([out_ch])
        w.update(wn(f"{base}.enc.cond_layer", t([2 * HIDDEN * WN_LAYERS, GIN, 1])))
        w[f"{base}.enc.cond_layer.bias"] = t([2 * HIDDEN * WN_LAYERS])
        w[f"{base}.post.weight"] = t([half, HIDDEN, 1], zero=True)
        w[f"{base}.post.bias"] = t([half], zero=True)

    # --- dec: GeneratorNSF ------------------------------------------------
    w["dec.conv_pre.weight"] = t([UPS_INIT, INTER, 7])
    w["dec.conv_pre.bias"] = t([UPS_INIT])
    w["dec.cond.weight"] = t([UPS_INIT, GIN, 1])
    w["dec.cond.bias"] = t([UPS_INIT])
    w["dec.m_source.l_linear.weight"] = t([1, HARMONICS + 1])
    w["dec.m_source.l_linear.bias"] = t([1])
    ch = UPS_INIT
    for i, (u, k) in enumerate(zip(UPS, UPS_K)):
        ch_out = UPS_INIT // (2 ** (i + 1))
        w.update(wn(f"dec.ups.{i}", t([ch, ch_out, k])))
        w[f"dec.ups.{i}.bias"] = t([ch_out])
        # noise_convs bridge the NSF source into each upsample stage
        stride_f0 = 1
        for r in UPS[i + 1 :]:
            stride_f0 *= r
        nk = stride_f0 * 2 - stride_f0 % 2 if stride_f0 > 1 else 1
        w[f"dec.noise_convs.{i}.weight"] = t([ch_out, 1, nk])
        w[f"dec.noise_convs.{i}.bias"] = t([ch_out])
        for j, (rk, rd) in enumerate(zip(RES_K, RES_D)):
            r = f"dec.resblocks.{i * len(RES_K) + j}"
            for c in range(len(rd)):
                w.update(wn(f"{r}.convs1.{c}", t([ch_out, ch_out, rk])))
                w[f"{r}.convs1.{c}.bias"] = t([ch_out])
                w.update(wn(f"{r}.convs2.{c}", t([ch_out, ch_out, rk])))
                w[f"{r}.convs2.{c}.bias"] = t([ch_out])
        ch = ch_out
    # conv_post has bias=False in RVC, so no bias entry exists.
    w["dec.conv_post.weight"] = t([1, ch, 7])

    # --- speaker embedding -----------------------------------------------
    w["emb_g.weight"] = t([SPK, GIN])

    return w


def write_pth(path, checkpoint):
    """Write a torch-layout ZIP: archive/data.pkl + archive/data/<key>."""
    tensors = []

    def collect(obj):
        if isinstance(obj, FakeTensor):
            tensors.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                collect(v)

    collect(checkpoint)

    buf = io.BytesIO()
    p = _Pickler(buf, protocol=2)
    p.dump(checkpoint)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        for tensor in tensors:
            z.writestr(f"archive/data/{tensor.key}", tensor.storage_bytes())


def main():
    out_dir = Path(__file__).parent

    global _seed
    _seed = 0x12345678
    FakeTensor._next_key = 0
    write_pth(
        out_dir / "tiny_v2_f0.pth",
        {
            "config": CONFIG,
            "weight": build_weights(),
            "f0": 1,
            "version": "v2",
            "info": "tiny synthetic fixture for vc-convert tests",
        },
    )

    FakeTensor._next_key = 0
    write_pth(
        out_dir / "tiny_v1.pth",
        {
            "config": CONFIG,
            "weight": {"emb_g.weight": t([SPK, GIN])},
            "f0": 1,
            "version": "v1",
        },
    )

    FakeTensor._next_key = 0
    write_pth(
        out_dir / "tiny_v2_nof0.pth",
        {
            "config": CONFIG,
            "weight": {"emb_g.weight": t([SPK, GIN])},
            "f0": 0,
            "version": "v2",
        },
    )

    for name in ("tiny_v2_f0.pth", "tiny_v1.pth", "tiny_v2_nof0.pth"):
        size = (out_dir / name).stat().st_size
        print(f"{name}: {size} bytes")


if __name__ == "__main__":
    main()
