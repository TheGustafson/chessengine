// Shape-faithful microbenchmark of ARCHON's NNUE head kernels (perf study 46).
// Calls are CHAINED (output feeds the next call's input) because forward_into is
// a strictly serial pipeline -- so this measures the kernel's LATENCY, which is
// the quantity a dependency-chain question is about.
#![allow(internal_features)]
use std::arch::x86_64::*;
use std::hint::black_box;

const L2: usize = 16;
const L3: usize = 32;

// ---------------- l2_tail: the 16x32 f32 matvec -------------------------
#[target_feature(enable = "avx512f")]
unsafe fn l2_tail_2(out: &mut [f32; L3], l1: &[f32; L2], w: &[f32; L2 * L3], b: &[f32; L3]) {
    unsafe {
        let mut a0 = _mm512_loadu_ps(b.as_ptr());
        let mut a1 = _mm512_loadu_ps(b.as_ptr().add(16));
        for (i, &input) in l1.iter().enumerate() {
            let li = _mm512_set1_ps(input);
            let row = w.as_ptr().add(i * L3);
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(row), li, a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(row.add(16)), li, a1);
        }
        let z = _mm512_setzero_ps(); let one = _mm512_set1_ps(1.0);
        let s = |a| { let c = _mm512_min_ps(_mm512_max_ps(a, z), one); _mm512_mul_ps(c, c) };
        _mm512_storeu_ps(out.as_mut_ptr(), s(a0));
        _mm512_storeu_ps(out.as_mut_ptr().add(16), s(a1));
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_tail_4(out: &mut [f32; L3], l1: &[f32; L2], w: &[f32; L2 * L3], b: &[f32; L3]) {
    unsafe {
        let mut a0 = _mm512_loadu_ps(b.as_ptr());
        let mut a1 = _mm512_loadu_ps(b.as_ptr().add(16));
        let mut c0 = _mm512_setzero_ps();
        let mut c1 = _mm512_setzero_ps();
        let mut i = 0;
        while i < L2 {
            let li = _mm512_set1_ps(l1[i]);
            let row = w.as_ptr().add(i * L3);
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(row), li, a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(row.add(16)), li, a1);
            let lj = _mm512_set1_ps(l1[i + 1]);
            let row2 = w.as_ptr().add((i + 1) * L3);
            c0 = _mm512_fmadd_ps(_mm512_loadu_ps(row2), lj, c0);
            c1 = _mm512_fmadd_ps(_mm512_loadu_ps(row2.add(16)), lj, c1);
            i += 2;
        }
        let a0 = _mm512_add_ps(a0, c0);
        let a1 = _mm512_add_ps(a1, c1);
        let z = _mm512_setzero_ps(); let one = _mm512_set1_ps(1.0);
        let s = |a| { let c = _mm512_min_ps(_mm512_max_ps(a, z), one); _mm512_mul_ps(c, c) };
        _mm512_storeu_ps(out.as_mut_ptr(), s(a0));
        _mm512_storeu_ps(out.as_mut_ptr().add(16), s(a1));
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_tail_8(out: &mut [f32; L3], l1: &[f32; L2], w: &[f32; L2 * L3], b: &[f32; L3]) {
    unsafe {
        let mut a = [_mm512_setzero_ps(); 8];
        a[0] = _mm512_loadu_ps(b.as_ptr());
        a[1] = _mm512_loadu_ps(b.as_ptr().add(16));
        let mut i = 0;
        while i < L2 {
            for k in 0..4 {
                let li = _mm512_set1_ps(l1[i + k]);
                let row = w.as_ptr().add((i + k) * L3);
                a[2 * k] = _mm512_fmadd_ps(_mm512_loadu_ps(row), li, a[2 * k]);
                a[2 * k + 1] = _mm512_fmadd_ps(_mm512_loadu_ps(row.add(16)), li, a[2 * k + 1]);
            }
            i += 4;
        }
        let a0 = _mm512_add_ps(_mm512_add_ps(a[0], a[2]), _mm512_add_ps(a[4], a[6]));
        let a1 = _mm512_add_ps(_mm512_add_ps(a[1], a[3]), _mm512_add_ps(a[5], a[7]));
        let z = _mm512_setzero_ps(); let one = _mm512_set1_ps(1.0);
        let s = |x| { let c = _mm512_min_ps(_mm512_max_ps(x, z), one); _mm512_mul_ps(c, c) };
        _mm512_storeu_ps(out.as_mut_ptr(), s(a0));
        _mm512_storeu_ps(out.as_mut_ptr().add(16), s(a1));
    }
}

// ---------------- l1_dot: CHAINS=4 over `count` active groups -----------
const ACC: usize = 1280;
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn l1_dot_cur(pre: &mut [i32; 16], ft: &[u8; ACC], nnz: &[u16], w: &[i8]) {
    unsafe {
        let mut sums = [_mm512_setzero_si512(); 4];
        sums[0] = _mm512_loadu_si512(pre.as_ptr().cast());
        let (quads, tail) = nnz.as_chunks::<4>();
        for quad in quads {
            for (chain, &g) in quad.iter().enumerate() {
                let g = usize::from(g);
                let inputs = _mm512_set1_epi32(i32::from_le_bytes([
                    ft[g * 4], ft[g * 4 + 1], ft[g * 4 + 2], ft[g * 4 + 3]]));
                let rows = _mm512_loadu_si512(w.as_ptr().add(g * 64).cast());
                sums[chain] = _mm512_dpbusd_epi32(sums[chain], inputs, rows);
            }
        }
        // THE CURRENT SHAPE: every tail group folds into chain 0.
        for &g in tail {
            let g = usize::from(g);
            let inputs = _mm512_set1_epi32(i32::from_le_bytes([
                ft[g * 4], ft[g * 4 + 1], ft[g * 4 + 2], ft[g * 4 + 3]]));
            let rows = _mm512_loadu_si512(w.as_ptr().add(g * 64).cast());
            sums[0] = _mm512_dpbusd_epi32(sums[0], inputs, rows);
        }
        let t = _mm512_add_epi32(_mm512_add_epi32(sums[0], sums[1]),
                                 _mm512_add_epi32(sums[2], sums[3]));
        _mm512_storeu_si512(pre.as_mut_ptr().cast(), t);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn l1_dot_spread(pre: &mut [i32; 16], ft: &[u8; ACC], nnz: &[u16], w: &[i8]) {
    unsafe {
        let mut sums = [_mm512_setzero_si512(); 4];
        sums[0] = _mm512_loadu_si512(pre.as_ptr().cast());
        let (quads, tail) = nnz.as_chunks::<4>();
        for quad in quads {
            for (chain, &g) in quad.iter().enumerate() {
                let g = usize::from(g);
                let inputs = _mm512_set1_epi32(i32::from_le_bytes([
                    ft[g * 4], ft[g * 4 + 1], ft[g * 4 + 2], ft[g * 4 + 3]]));
                let rows = _mm512_loadu_si512(w.as_ptr().add(g * 64).cast());
                sums[chain] = _mm512_dpbusd_epi32(sums[chain], inputs, rows);
            }
        }
        // THE FIX: tail group k folds into chain k -- still i32 adds, so the
        // VALUE is identical (wrapping i32 addition is associative).
        for (chain, &g) in tail.iter().enumerate() {
            let g = usize::from(g);
            let inputs = _mm512_set1_epi32(i32::from_le_bytes([
                ft[g * 4], ft[g * 4 + 1], ft[g * 4 + 2], ft[g * 4 + 3]]));
            let rows = _mm512_loadu_si512(w.as_ptr().add(g * 64).cast());
            sums[chain] = _mm512_dpbusd_epi32(sums[chain], inputs, rows);
        }
        let t = _mm512_add_epi32(_mm512_add_epi32(sums[0], sums[1]),
                                 _mm512_add_epi32(sums[2], sums[3]));
        _mm512_storeu_si512(pre.as_mut_ptr().cast(), t);
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let which = a[1].clone();
    let iters: u64 = a[2].parse().unwrap();
    let count: usize = if a.len() > 3 { a[3].parse().unwrap() } else { 39 };

    let mut w2 = [0.0f32; L2 * L3];
    for (i, v) in w2.iter_mut().enumerate() { *v = ((i % 17) as f32 - 8.0) / 64.0; }
    let b2 = [0.01f32; L3];
    let mut l1 = [0.5f32; L2];
    let mut out = [0.0f32; L3];

    let mut ft = [0u8; ACC];
    for (i, v) in ft.iter_mut().enumerate() { *v = ((i * 37) % 251) as u8; }
    let w1 = vec![3i8; 320 * 64];
    let nnz: Vec<u16> = (0..count).map(|k| ((k * 7 + 3) % 320) as u16).collect();
    let mut pre = [1i32; 16];

    unsafe {
        match which.as_str() {
            "l2tail2" => for _ in 0..iters { l2_tail_2(&mut out, &l1, &w2, &b2); l1.copy_from_slice(&out[..16]); black_box(&mut l1); },
            "l2tail4" => for _ in 0..iters { l2_tail_4(&mut out, &l1, &w2, &b2); l1.copy_from_slice(&out[..16]); black_box(&mut l1); },
            "l2tail8" => for _ in 0..iters { l2_tail_8(&mut out, &l1, &w2, &b2); l1.copy_from_slice(&out[..16]); black_box(&mut l1); },
            "l1dotcur" => for _ in 0..iters { l1_dot_cur(&mut pre, &ft, &nnz, &w1); black_box(&mut pre); },
            "l1dotspread" => for _ in 0..iters { l1_dot_spread(&mut pre, &ft, &nnz, &w1); black_box(&mut pre); },
            _ => { eprintln!("unknown"); std::process::exit(2); }
        }
    }
    black_box(&mut w2); black_box(&mut out); black_box(&mut pre);
    println!("{}", iters);
}
