#!/usr/bin/env python3
import struct, sys
import numpy as np

def parse_raw(path):
    """Parse bullet ModelWeights.write_into format: repeated [id\n][u64 len][len f32]."""
    with open(path,'rb') as f:
        buf=f.read()
    off=0; tensors={}
    while off < len(buf):
        nl=buf.index(b'\n', off)
        idn=buf[off:nl].decode()
        ln=struct.unpack_from('<Q', buf, nl+1)[0]
        start=nl+1+8
        arr=np.frombuffer(buf, dtype='<f4', count=ln, offset=start)
        tensors[idn]=arr
        off=start+ln*4
    return tensors

# quantization scheme per tensor: (scale, dtype_bits, signed_max)
SCHEME = {
    'l0_pstw':  (255, 'i16', 32767),
    'l0_pstb':  (255, 'i16', 32767),
    'l0_threatw':(64, 'i8',  127),
    'l1w':      (64,  'i8',  127),
}

def analyze(path, label):
    t=parse_raw(path)
    print(f"\n===== {label} =====")
    print(f"  file={path}")
    print(f"  tensors: {[(k,len(v)) for k,v in t.items()]}")
    for name,(scale,dt,smax) in SCHEME.items():
        w=t[name].astype(np.float64)
        q=np.round(w*scale)  # bullet uses round=True
        absq=np.abs(q)
        n=len(w)
        over = absq > smax        # would overflow -> bullet would ERROR
        n_over=int(over.sum())
        # also count how close to the boundary (within 1 LSB of clip)
        near = absq >= (smax)     # at or above representable positive max magnitude
        # note i8 min is -128; treat symmetric smax for "how much headroom" reporting
        maxq=float(absq.max())
        maxw=float(np.abs(w).max())
        limit_w = smax/scale
        print(f"  [{name}] scale={scale} {dt} |range|<= {smax} (== |w|<= {limit_w:.4f})")
        print(f"      n={n:,}  max|w|={maxw:.5f}  max|round(w*s)|={maxq:.0f}")
        print(f"      OVERFLOW (|q|>{smax}): {n_over:,}  ({100.0*n_over/n:.4f}%)")
        if n_over:
            worst=w[over]
            print(f"      overflow |w| range: {np.abs(worst).min():.4f}..{np.abs(worst).max():.4f}")
        # headroom histogram: fraction within X% of clip
        for frac in (0.90, 0.95, 0.99):
            thr=frac*limit_w
            c=int((np.abs(w)>=thr).sum())
            print(f"      |w|>={frac:.2f}*limit ({thr:.4f}): {c:,} ({100.0*c/n:.4f}%)")
    return t

t720=analyze("data/nnue/experiments/d40-long-v1/ckpt/d40-uho-720sb-final-raw.bin","720sb (FAILED runtime max 0.2517)")
t240=analyze("data/nnue/experiments/d40-uho-v1/ckpt/d40-uho-v1-240sb-final-raw.bin","240sb (PASSED runtime max 0.174)")

# Compare magnitude growth of the two i8 tensors between schedules.
print("\n===== schedule growth (i8 tensors) =====")
for name in ('l0_threatw','l1w'):
    a=np.abs(t240[name]); b=np.abs(t720[name])
    print(f"  {name}: max 240={a.max():.4f} 720={b.max():.4f} | "
          f"mean 240={a.mean():.5f} 720={b.mean():.5f} | "
          f"p99.9 240={np.percentile(a,99.9):.4f} 720={np.percentile(b,99.9):.4f}")
