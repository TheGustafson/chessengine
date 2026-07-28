   70530:	test   %rcx,%rcx
   70533:	je     708a2 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x372>
   70539:	add    %rcx,%rcx
   7053c:	vmovdqu (%rdi),%ymm1
   70540:	vmovdqu 0x20(%rdi),%ymm2
   70545:	xor    %r9d,%r9d
   70548:	vpbroadcastd -0x696e1(%rip),%ymm0        # 6e70 <__abi_tag+0x6b74>
   70551:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   70560:	movzwl (%rdx,%r9,1),%eax
   70565:	cmp    $0x140,%eax
   7056a:	jae    708a6 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x376>
   70570:	vpbroadcastd (%rsi,%rax,4),%ymm3
   70576:	shl    $0x9,%eax
   70579:	vpmaddubsw (%r8,%rax,1),%ymm3,%ymm4
   7057f:	vpmaddubsw 0x20(%r8,%rax,1),%ymm3,%ymm3
   70586:	vpmaddwd %ymm0,%ymm4,%ymm4
   7058a:	vpaddd %ymm1,%ymm4,%ymm1
   7058e:	vpmaddwd %ymm0,%ymm3,%ymm3
   70592:	vpaddd %ymm2,%ymm3,%ymm2
   70596:	add    $0x2,%r9
   7059a:	cmp    %r9,%rcx
   7059d:	jne    70560 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x30>
   7059f:	vmovdqu %ymm1,(%rdi)
   705a3:	vmovdqu %ymm2,0x20(%rdi)
   705a8:	vmovdqu 0x40(%rdi),%ymm1
   705ad:	vmovdqu 0x60(%rdi),%ymm2
   705b2:	xor    %r9d,%r9d
   705b5:	data16 cs nopw 0x0(%rax,%rax,1)
   705c0:	movzwl (%rdx,%r9,1),%eax
   705c5:	cmp    $0x13f,%eax
   705ca:	ja     708a6 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x376>
   705d0:	vpbroadcastd (%rsi,%rax,4),%ymm3
   705d6:	shl    $0x9,%eax
   705d9:	vpmaddubsw 0x40(%r8,%rax,1),%ymm3,%ymm4
   705e0:	vpmaddubsw 0x60(%r8,%rax,1),%ymm3,%ymm3
   705e7:	vpmaddwd %ymm0,%ymm4,%ymm4
   705eb:	vpaddd %ymm1,%ymm4,%ymm1
   705ef:	vpmaddwd %ymm0,%ymm3,%ymm3
   705f3:	vpaddd %ymm2,%ymm3,%ymm2
   705f7:	add    $0x2,%r9
   705fb:	cmp    %r9,%rcx
   705fe:	jne    705c0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x90>
   70600:	vmovdqu %ymm1,0x40(%rdi)
   70605:	vmovdqu %ymm2,0x60(%rdi)
   7060a:	vmovdqu 0x80(%rdi),%ymm1
   70612:	vmovdqu 0xa0(%rdi),%ymm2
   7061a:	xor    %r9d,%r9d
   7061d:	nopl   (%rax)
   70620:	movzwl (%rdx,%r9,1),%eax
   70625:	cmp    $0x13f,%eax
   7062a:	ja     708a6 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x376>
   70630:	vpbroadcastd (%rsi,%rax,4),%ymm3
   70636:	shl    $0x9,%eax
   70639:	vpmaddubsw 0x80(%r8,%rax,1),%ymm3,%ymm4
   70643:	vpmaddubsw 0xa0(%r8,%rax,1),%ymm3,%ymm3
   7064d:	vpmaddwd %ymm0,%ymm4,%ymm4
   70651:	vpaddd %ymm1,%ymm4,%ymm1
   70655:	vpmaddwd %ymm0,%ymm3,%ymm3
   70659:	vpaddd %ymm2,%ymm3,%ymm2
   7065d:	add    $0x2,%r9
   70661:	cmp    %r9,%rcx
   70664:	jne    70620 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0xf0>
   70666:	vmovdqu %ymm1,0x80(%rdi)
   7066e:	vmovdqu %ymm2,0xa0(%rdi)
   70676:	vmovdqu 0xc0(%rdi),%ymm1
   7067e:	vmovdqu 0xe0(%rdi),%ymm2
   70686:	xor    %r9d,%r9d
   70689:	nopl   0x0(%rax)
   70690:	movzwl (%rdx,%r9,1),%eax
   70695:	cmp    $0x13f,%eax
   7069a:	ja     708a6 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x376>
   706a0:	vpbroadcastd (%rsi,%rax,4),%ymm3
   706a6:	shl    $0x9,%eax
   706a9:	vpmaddubsw 0xc0(%r8,%rax,1),%ymm3,%ymm4
   706b3:	vpmaddubsw 0xe0(%r8,%rax,1),%ymm3,%ymm3
   706bd:	vpmaddwd %ymm0,%ymm4,%ymm4
   706c1:	vpaddd %ymm1,%ymm4,%ymm1
   706c5:	vpmaddwd %ymm0,%ymm3,%ymm3
   706c9:	vpaddd %ymm2,%ymm3,%ymm2
   706cd:	add    $0x2,%r9
   706d1:	cmp    %r9,%rcx
   706d4:	jne    70690 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x160>
   706d6:	vmovdqu %ymm1,0xc0(%rdi)
   706de:	vmovdqu %ymm2,0xe0(%rdi)
   706e6:	vmovdqu 0x100(%rdi),%ymm1
   706ee:	vmovdqu 0x120(%rdi),%ymm2
   706f6:	xor    %r9d,%r9d
   706f9:	nopl   0x0(%rax)
   70700:	movzwl (%rdx,%r9,1),%eax
   70705:	cmp    $0x13f,%eax
   7070a:	ja     708a6 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x376>
   70710:	vpbroadcastd (%rsi,%rax,4),%ymm3
   70716:	shl    $0x9,%eax
   70719:	vpmaddubsw 0x100(%r8,%rax,1),%ymm3,%ymm4
   70723:	vpmaddubsw 0x120(%r8,%rax,1),%ymm3,%ymm3
   7072d:	vpmaddwd %ymm0,%ymm4,%ymm4
   70731:	vpaddd %ymm1,%ymm4,%ymm1
   70735:	vpmaddwd %ymm0,%ymm3,%ymm3
   70739:	vpaddd %ymm2,%ymm3,%ymm2
   7073d:	add    $0x2,%r9
   70741:	cmp    %r9,%rcx
   70744:	jne    70700 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x1d0>
   70746:	vmovdqu %ymm1,0x100(%rdi)
   7074e:	vmovdqu %ymm2,0x120(%rdi)
   70756:	vmovdqu 0x140(%rdi),%ymm1
   7075e:	vmovdqu 0x160(%rdi),%ymm2
   70766:	xor    %r9d,%r9d
   70769:	nopl   0x0(%rax)
   70770:	movzwl (%rdx,%r9,1),%eax
   70775:	cmp    $0x13f,%eax
   7077a:	ja     708a6 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x376>
   70780:	vpbroadcastd (%rsi,%rax,4),%ymm3
   70786:	shl    $0x9,%eax
   70789:	vpmaddubsw 0x140(%r8,%rax,1),%ymm3,%ymm4
   70793:	vpmaddubsw 0x160(%r8,%rax,1),%ymm3,%ymm3
   7079d:	vpmaddwd %ymm0,%ymm4,%ymm4
   707a1:	vpaddd %ymm1,%ymm4,%ymm1
   707a5:	vpmaddwd %ymm0,%ymm3,%ymm3
   707a9:	vpaddd %ymm2,%ymm3,%ymm2
   707ad:	add    $0x2,%r9
   707b1:	cmp    %r9,%rcx
   707b4:	jne    70770 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x240>
   707b6:	vmovdqu %ymm1,0x140(%rdi)
   707be:	vmovdqu %ymm2,0x160(%rdi)
   707c6:	vmovdqu 0x180(%rdi),%ymm1
   707ce:	vmovdqu 0x1a0(%rdi),%ymm2
   707d6:	xor    %r9d,%r9d
   707d9:	nopl   0x0(%rax)
   707e0:	movzwl (%rdx,%r9,1),%eax
   707e5:	cmp    $0x13f,%eax
   707ea:	ja     708a6 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x376>
   707f0:	vpbroadcastd (%rsi,%rax,4),%ymm3
   707f6:	shl    $0x9,%eax
   707f9:	vpmaddubsw 0x180(%r8,%rax,1),%ymm3,%ymm4
   70803:	vpmaddubsw 0x1a0(%r8,%rax,1),%ymm3,%ymm3
   7080d:	vpmaddwd %ymm0,%ymm4,%ymm4
   70811:	vpaddd %ymm1,%ymm4,%ymm1
   70815:	vpmaddwd %ymm0,%ymm3,%ymm3
   70819:	vpaddd %ymm2,%ymm3,%ymm2
   7081d:	add    $0x2,%r9
   70821:	cmp    %r9,%rcx
   70824:	jne    707e0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x2b0>
   70826:	vmovdqu %ymm1,0x180(%rdi)
   7082e:	vmovdqu %ymm2,0x1a0(%rdi)
   70836:	vmovdqu 0x1c0(%rdi),%ymm1
   7083e:	vmovdqu 0x1e0(%rdi),%ymm2
   70846:	xor    %r9d,%r9d
   70849:	nopl   0x0(%rax)
   70850:	movzwl (%rdx,%r9,1),%eax
   70855:	cmp    $0x13f,%eax
   7085a:	ja     708a6 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x376>
   7085c:	vpbroadcastd (%rsi,%rax,4),%ymm3
   70862:	shl    $0x9,%eax
   70865:	vpmaddubsw 0x1c0(%r8,%rax,1),%ymm3,%ymm4
   7086f:	vpmaddubsw 0x1e0(%r8,%rax,1),%ymm3,%ymm3
   70879:	vpmaddwd %ymm0,%ymm4,%ymm4
   7087d:	vpaddd %ymm1,%ymm4,%ymm1
   70881:	vpmaddwd %ymm0,%ymm3,%ymm3
   70885:	vpaddd %ymm2,%ymm3,%ymm2
   70889:	add    $0x2,%r9
   7088d:	cmp    %r9,%rcx
   70890:	jne    70850 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd4avx26l1_dot+0x320>
   70892:	vmovdqu %ymm1,0x1c0(%rdi)
   7089a:	vmovdqu %ymm2,0x1e0(%rdi)
   708a2:	vzeroupper
   708a5:	ret
   708a6:	shl    $0x2,%eax
   708a9:	push   %rax
   708aa:	lea    0x69157(%rip),%rdx        # d9a08 <__frame_dummy_init_array_entry+0x2288>
   708b1:	mov    $0x500,%esi
   708b6:	mov    %rax,%rdi
   708b9:	vzeroupper
   708bc:	call   9406b <_RNvNtCs4NRVxsYgnAr_4core9panicking18panic_bounds_check>
