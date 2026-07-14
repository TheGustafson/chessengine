"""Regression tests for the Bullet-quantized → CENN-v2 D40 byte boundary."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "netpack", Path(__file__).with_name("netpack.py")
)
netpack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(netpack)


class D40PackingTests(unittest.TestCase):
    def test_feature_transformers_remain_feature_major(self):
        # Shrink the architecture so a signature checkpoint exercises the real
        # packer without allocating the production 81 MB tensors.
        netpack.HALFKA_DIMS = 3
        netpack.THREAT_DIMS = 2
        netpack.D40_ACC = 4
        netpack.D40_L2 = 2
        netpack.D40_L3 = 2
        netpack.D40_BUCKETS = 2

        pst = np.arange(1, 13, dtype="<i2").reshape(3, 4)
        bias = np.arange(21, 25, dtype="<i2")
        threat = np.arange(31, 39, dtype="i1").reshape(2, 4)
        l1w = np.arange(41, 57, dtype="i1").reshape(4, 4)
        l1b = np.arange(61, 65, dtype="<f4")
        l2w = np.arange(71, 79, dtype="<f4").reshape(4, 2)
        l2b = np.arange(81, 85, dtype="<f4")
        l3w = np.arange(91, 95, dtype="<f4").reshape(2, 2)
        l3b = np.arange(101, 103, dtype="<f4")
        tensors = [pst, bias, threat, l1w, l1b, l2w, l2b, l3w, l3b]
        raw = b"".join(tensor.tobytes() for tensor in tensors)
        raw += bytes(b"bullet"[i % 6] for i in range((-len(raw)) % 64))

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "quant.bin"
            packed = Path(directory) / "net.cenn"
            source.write_bytes(raw)
            netpack.pack_d40(str(source), str(packed))
            payload = packed.read_bytes()[netpack.HEADER_V2_LEN :]

        offset = 0
        pst_bytes = pst.nbytes
        self.assertEqual(payload[offset : offset + pst_bytes], pst.tobytes())
        offset += pst_bytes + bias.nbytes
        self.assertEqual(payload[offset : offset + threat.nbytes], threat.tobytes())


if __name__ == "__main__":
    unittest.main()
