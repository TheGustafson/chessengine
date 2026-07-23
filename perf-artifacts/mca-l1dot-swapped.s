# Proposed group-outer l1_dot loop body: one iteration = one nnz group
# folded into ALL 8 output blocks (zmm0-zmm7 live accumulators).
# nnz walked ONCE; broadcast formed once per group; 8 independent dpbusd.
movzwl (%rdx,%r9,1),%eax
mov    %eax,%r10d
shl    $0x9,%r10d
vpbroadcastd (%rsi,%rax,4),%zmm8
add    $0x2,%r9
vpdpbusd (%r8,%r10,1),%zmm8,%zmm0
vpdpbusd 0x40(%r8,%r10,1),%zmm8,%zmm1
vpdpbusd 0x80(%r8,%r10,1),%zmm8,%zmm2
vpdpbusd 0xc0(%r8,%r10,1),%zmm8,%zmm3
vpdpbusd 0x100(%r8,%r10,1),%zmm8,%zmm4
vpdpbusd 0x140(%r8,%r10,1),%zmm8,%zmm5
vpdpbusd 0x180(%r8,%r10,1),%zmm8,%zmm6
vpdpbusd 0x1c0(%r8,%r10,1),%zmm8,%zmm7
cmp    %r9,%rcx
