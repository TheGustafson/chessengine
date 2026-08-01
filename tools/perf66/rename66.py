#!/usr/bin/env python3
"""perf study 66 — one-shot rename of study 65's `threat_narrow` bitmap into
the study-66 `threat_rowmap` byte-offset map. Kept in-tree only as the receipt
for a mechanical edit; it is not part of any build."""

EDITS = {
    'src/eval/nnue/sota_loader.rs': [
        ("""    let mut threat_weights = if has_threats {
        read_i8s_aligned(payload, &mut cur, threat_rows * ACC)
    } else {
        Aligned::from_iter_len(0, std::iter::empty())
    };
    // Perf study 65 — pack the narrow rows in place. See `threat_narrow`.
    let threat_narrow = pack_narrow_plane(&mut threat_weights);""",
         """    let threat_weights = if has_threats {
        read_i8s_aligned(payload, &mut cur, threat_rows * ACC)
    } else {
        Aligned::from_iter_len(0, std::iter::empty())
    };
    // Perf studies 65/66 — pack the narrow rows. See `threat_rowmap`.
    let (threat_weights, threat_rowmap) = pack_threat_plane(threat_weights);"""),
        ("        threat_narrow,\n", "        threat_rowmap,\n"),
        ("            threat_narrow: Vec::new(),", "            threat_rowmap: Vec::new(),"),
        ("""        for (r, row) in net.threat_weights.chunks_exact(ACC).enumerate() {
            if super::sota_simd::is_narrow(&net.threat_narrow, r * ACC) {
                for j in 0..ACC {
                    payload.push(super::sota_simd::unpack_narrow_lane(row, j) as i8 as u8);
                }
            } else {
                for &w in row {
                    payload.push(w as u8);
                }
            }
        }""",
         """        let rows = if net.threat_rowmap.is_empty() {
            net.threat_weights.len() / ACC
        } else {
            net.threat_rowmap.len()
        };
        for r in 0..rows {
            let (base, narrow) = super::sota_simd::row_entry(&net.threat_rowmap, r * ACC);
            if narrow {
                let row = &net.threat_weights[base..base + super::sota_simd::NARROW_BYTES];
                for j in 0..ACC {
                    payload.push(super::sota_simd::unpack_narrow_lane(row, j) as i8 as u8);
                }
            } else {
                for &w in &net.threat_weights[base..base + ACC] {
                    payload.push(w as u8);
                }
            }
        }"""),
        ("                &compressed.threat_narrow,", "                &compressed.threat_rowmap,"),
        ("                &scattered.threat_narrow,", "                &scattered.threat_rowmap,"),
    ],
    'src/eval/nnue/sota_context.rs': [
        (".with_narrow(&large.threat_narrow)", ".with_narrow(&large.threat_rowmap)"),
        (".with_narrow(&net.threat_narrow)", ".with_narrow(&net.threat_rowmap)"),
        (".with_narrow(&largenet.threat_narrow)", ".with_narrow(&largenet.threat_rowmap)"),
        ("            &self.net.threat_narrow,", "            &self.net.threat_rowmap,"),
    ],
    'src/eval/nnue/sota_features.rs': [
        ("    threat_narrow: &[u64],", "    threat_rowmap: &[u32],"),
        ("        threat_narrow,\n", "        threat_rowmap,\n"),
        ("            threat_narrow,\n", "            threat_rowmap,\n"),
        ("add_threat_column(dst, threat_weights, threat_narrow, idx)",
         "add_threat_column(dst, threat_weights, threat_rowmap, idx)"),
        ("""fn add_threat_column(acc: &mut [i16; ACC], weights: &[i8], narrow: &[u64], index: usize) {
    let base = index * ACC;
    if super::sota_simd::is_narrow(narrow, base) {
        let row = &weights[base..base + super::sota_simd::NARROW_BYTES];
        for (j, a) in acc.iter_mut().enumerate() {
            *a = a.wrapping_add(super::sota_simd::unpack_narrow_lane(row, j));
        }
    } else {
        add_column(acc, weights, index);
    }
}""",
         """fn add_threat_column(acc: &mut [i16; ACC], weights: &[i8], row_map: &[u32], index: usize) {
    let (base, narrow) = super::sota_simd::row_entry(row_map, index * ACC);
    if narrow {
        let row = &weights[base..base + super::sota_simd::NARROW_BYTES];
        for (j, a) in acc.iter_mut().enumerate() {
            *a = a.wrapping_add(super::sota_simd::unpack_narrow_lane(row, j));
        }
    } else {
        for (a, &w) in acc.iter_mut().zip(&weights[base..base + ACC]) {
            *a = a.wrapping_add(i16::from(w));
        }
    }
}"""),
    ],
}

for path, subs in EDITS.items():
    s = open(path).read()
    for a, b in subs:
        if s.count(a) == 0:
            print(f"MISS {path}: {a[:70]!r}")
        s = s.replace(a, b)
    open(path, 'w').write(s)
print("done")
