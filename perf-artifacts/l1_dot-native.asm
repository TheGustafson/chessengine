   71100:	test   %rcx,%rcx
   71103:	je     71312 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x212>
   71109:	vmovdqu64 (%rdi),%zmm0
   7110f:	add    %rcx,%rcx
   71112:	xor    %r9d,%r9d
   71115:	data16 cs nopw 0x0(%rax,%rax,1)
   71120:	movzwl (%rdx,%r9,1),%eax
   71125:	cmp    $0x140,%eax
   7112a:	jae    71316 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x216>
   71130:	mov    %eax,%r10d
   71133:	shl    $0x9,%r10d
   71137:	vpbroadcastd (%rsi,%rax,4),%zmm1
   7113e:	add    $0x2,%r9
   71142:	vpdpbusd (%r8,%r10,1),%zmm1,%zmm0
   71149:	cmp    %r9,%rcx
   7114c:	jne    71120 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x20>
   7114e:	vmovdqu64 %zmm0,(%rdi)
   71154:	xor    %r9d,%r9d
   71157:	vmovdqu64 0x40(%rdi),%zmm0
   7115e:	xchg   %ax,%ax
   71160:	movzwl (%rdx,%r9,1),%eax
   71165:	cmp    $0x13f,%eax
   7116a:	ja     71316 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x216>
   71170:	mov    %eax,%r10d
   71173:	shl    $0x9,%r10d
   71177:	vpbroadcastd (%rsi,%rax,4),%zmm1
   7117e:	add    $0x2,%r9
   71182:	vpdpbusd 0x40(%r8,%r10,1),%zmm1,%zmm0
   7118a:	cmp    %r9,%rcx
   7118d:	jne    71160 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x60>
   7118f:	vmovdqu64 %zmm0,0x40(%rdi)
   71196:	xor    %r9d,%r9d
   71199:	vmovdqu64 0x80(%rdi),%zmm0
   711a0:	movzwl (%rdx,%r9,1),%eax
   711a5:	cmp    $0x13f,%eax
   711aa:	ja     71316 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x216>
   711b0:	mov    %eax,%r10d
   711b3:	shl    $0x9,%r10d
   711b7:	vpbroadcastd (%rsi,%rax,4),%zmm1
   711be:	add    $0x2,%r9
   711c2:	vpdpbusd 0x80(%r8,%r10,1),%zmm1,%zmm0
   711ca:	cmp    %r9,%rcx
   711cd:	jne    711a0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0xa0>
   711cf:	vmovdqu64 %zmm0,0x80(%rdi)
   711d6:	xor    %r9d,%r9d
   711d9:	vmovdqu64 0xc0(%rdi),%zmm0
   711e0:	movzwl (%rdx,%r9,1),%eax
   711e5:	cmp    $0x13f,%eax
   711ea:	ja     71316 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x216>
   711f0:	mov    %eax,%r10d
   711f3:	shl    $0x9,%r10d
   711f7:	vpbroadcastd (%rsi,%rax,4),%zmm1
   711fe:	add    $0x2,%r9
   71202:	vpdpbusd 0xc0(%r8,%r10,1),%zmm1,%zmm0
   7120a:	cmp    %r9,%rcx
   7120d:	jne    711e0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0xe0>
   7120f:	vmovdqu64 %zmm0,0xc0(%rdi)
   71216:	xor    %r9d,%r9d
   71219:	vmovdqu64 0x100(%rdi),%zmm0
   71220:	movzwl (%rdx,%r9,1),%eax
   71225:	cmp    $0x13f,%eax
   7122a:	ja     71316 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x216>
   71230:	mov    %eax,%r10d
   71233:	shl    $0x9,%r10d
   71237:	vpbroadcastd (%rsi,%rax,4),%zmm1
   7123e:	add    $0x2,%r9
   71242:	vpdpbusd 0x100(%r8,%r10,1),%zmm1,%zmm0
   7124a:	cmp    %r9,%rcx
   7124d:	jne    71220 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x120>
   7124f:	vmovdqu64 %zmm0,0x100(%rdi)
   71256:	xor    %r9d,%r9d
   71259:	vmovdqu64 0x140(%rdi),%zmm0
   71260:	movzwl (%rdx,%r9,1),%eax
   71265:	cmp    $0x13f,%eax
   7126a:	ja     71316 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x216>
   71270:	mov    %eax,%r10d
   71273:	shl    $0x9,%r10d
   71277:	vpbroadcastd (%rsi,%rax,4),%zmm1
   7127e:	add    $0x2,%r9
   71282:	vpdpbusd 0x140(%r8,%r10,1),%zmm1,%zmm0
   7128a:	cmp    %r9,%rcx
   7128d:	jne    71260 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x160>
   7128f:	vmovdqu64 %zmm0,0x140(%rdi)
   71296:	xor    %r9d,%r9d
   71299:	vmovdqu64 0x180(%rdi),%zmm0
   712a0:	movzwl (%rdx,%r9,1),%eax
   712a5:	cmp    $0x13f,%eax
   712aa:	ja     71316 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x216>
   712ac:	mov    %eax,%r10d
   712af:	shl    $0x9,%r10d
   712b3:	vpbroadcastd (%rsi,%rax,4),%zmm1
   712ba:	add    $0x2,%r9
   712be:	vpdpbusd 0x180(%r8,%r10,1),%zmm1,%zmm0
   712c6:	cmp    %r9,%rcx
   712c9:	jne    712a0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x1a0>
   712cb:	vmovdqu64 %zmm0,0x180(%rdi)
   712d2:	xor    %r9d,%r9d
   712d5:	vmovdqu64 0x1c0(%rdi),%zmm0
   712dc:	nopl   0x0(%rax)
   712e0:	movzwl (%rdx,%r9,1),%eax
   712e5:	cmp    $0x13f,%eax
   712ea:	ja     71316 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x216>
   712ec:	mov    %eax,%r10d
   712ef:	shl    $0x9,%r10d
   712f3:	vpbroadcastd (%rsi,%rax,4),%zmm1
   712fa:	add    $0x2,%r9
   712fe:	vpdpbusd 0x1c0(%r8,%r10,1),%zmm1,%zmm0
   71306:	cmp    %r9,%rcx
   71309:	jne    712e0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx5126l1_dot+0x1e0>
   7130b:	vmovdqu64 %zmm0,0x1c0(%rdi)
   71312:	vzeroupper
   71315:	ret
   71316:	shl    $0x2,%eax
   71319:	push   %rax
   7131a:	lea    0x6966f(%rip),%rdx        # da990 <__frame_dummy_init_array_entry+0x21d0>
   71321:	mov    $0x500,%esi
   71326:	mov    %rax,%rdi
   71329:	vzeroupper
   7132c:	call   95a7b <_RNvNtCs4NRVxsYgnAr_4core9panicking18panic_bounds_check>
   71331:	int3
   71332:	int3
   71333:	int3
   71334:	int3
   71335:	int3
   71336:	int3
   71337:	int3
   71338:	int3
   71339:	int3
   7133a:	int3
   7133b:	int3
   7133c:	int3
   7133d:	int3
   7133e:	int3
   7133f:	int3
