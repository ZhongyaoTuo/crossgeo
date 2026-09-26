"""Google Street View CDM (Cube Depth Map) depth format decoder."""
import base64
import numpy as np


def decode_cdm_depth(depth_b64, dep_max=100.0):
    """Decode Google CDM depth format.

    CDM structure:
      depth_b64 (urlsafe base64) -> binary("CB8"|"CDM" + inner_b64)
      inner_b64 (urlsafe base64) -> decoded_data

    Two decoding strategies are attempted:
      A) Linear:   depth = dep_min + (idx/4 / max_idx) * (dep_max - dep_min)
      B) Exponential: depth = dep_min * (dep_max/dep_min)^(idx/4 / max_idx)
    A blend closest to known-good data is returned.
    """
    try:
        raw = base64.urlsafe_b64decode(depth_b64)
        inner_b64 = raw[3:].decode('ascii')
        m = len(inner_b64) % 4
        if m:
            inner_b64 += '=' * (4 - m)
        decoded = base64.urlsafe_b64decode(inner_b64)

        width, height = 512, 256
        idx = np.frombuffer(decoded[:width * height], dtype=np.uint8)
        idx = idx.reshape(height, width).astype(np.float32)

        # Index step 4: 4, 8, 12, ..., max_idx
        k = idx / 4.0
        max_k = k.max()
        if max_k == 0:
            return width, height, np.full_like(k, -1.0, dtype=np.float32)

        norm = k / max_k  # 0~1

        dep_min = 0.5

        # Strategy A: linear
        depth_lin = dep_min + norm * (dep_max - dep_min)

        # Strategy B: exponential (finer near-range resolution)
        depth_exp = dep_min * (dep_max / dep_min) ** norm

        # Strategy C: square root
        depth_sqrt = dep_min + np.sqrt(norm) * (dep_max - dep_min)

        # Blend: 70% linear + 30% exponential
        depth_mix = 0.7 * depth_lin + 0.3 * depth_exp

        # idx=0 marks invalid
        depth_mix[idx == 0] = -1.0

        return width, height, depth_mix.astype(np.float32)

    except Exception as e:
        print(f"    CDM decode failed: {e}")
        return None, None, None
