# Current l1_dot inner loop body (one of the 8 unrolled block loops),
# transcribed from target/release/chessengine objdump at 0x71120.
# One iteration = one nnz group folded into ONE output block.
# The full eval runs this loop 8 times (once per block), re-walking nnz.
movzwl (%rdx,%r9,1),%eax
mov    %eax,%r10d
shl    $0x9,%r10d
vpbroadcastd (%rsi,%rax,4),%zmm1
add    $0x2,%r9
vpdpbusd (%r8,%r10,1),%zmm1,%zmm0
cmp    %r9,%rcx
