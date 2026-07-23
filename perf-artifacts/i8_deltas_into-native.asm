   70af0:	push   %rbp
   70af1:	push   %r15
   70af3:	push   %r14
   70af5:	push   %r13
   70af7:	push   %r12
   70af9:	push   %rbx
   70afa:	mov    0x40(%rsp),%rcx
   70aff:	mov    0x38(%rsp),%rax
   70b04:	lea    0x0(,%rcx,8),%r12
   70b0c:	lea    (%rax,%rcx,8),%r10
   70b10:	test   %r9,%r9
   70b13:	je     70de9 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x2f9>
   70b19:	lea    (%r8,%r9,8),%r11
   70b1d:	shl    $0x3,%r9
   70b21:	mov    %r8,-0x20(%rsp)
   70b26:	add    $0xfffffffffffffff8,%r9
   70b2a:	mov    %r9,%r8
   70b2d:	shr    $0x3,%r8
   70b31:	inc    %r8
   70b34:	test   %rcx,%rcx
   70b37:	je     70f69 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x479>
   70b3d:	add    $0xfffffffffffffff8,%r12
   70b41:	and    $0x7,%r8d
   70b45:	mov    %r9,%r15
   70b48:	mov    %r9,-0x8(%rsp)
   70b4d:	and    $0x38,%r15d
   70b51:	mov    $0x28,%r13d
   70b57:	xor    %ebp,%ebp
   70b59:	mov    %r12d,%ecx
   70b5c:	shr    $0x3,%ecx
   70b5f:	mov    %r12,-0x10(%rsp)
   70b64:	neg    %r8
   70b67:	and    $0x38,%r12d
   70b6b:	inc    %ecx
   70b6d:	mov    %r8,-0x18(%rsp)
   70b72:	and    $0x7,%ecx
   70b75:	neg    %rcx
   70b78:	mov    %rcx,%r9
   70b7b:	jmp    70b94 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0xa4>
   70b7d:	nopl   (%rax)
   70b80:	vmovdqu64 %zmm0,(%rdi,%rbp,2)
   70b87:	add    $0x20,%rbp
   70b8b:	dec    %r13
   70b8e:	je     710d9 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x5e9>
   70b94:	vmovdqu64 (%rsi,%rbp,2),%zmm0
   70b9b:	lea    (%rdx,%rbp,1),%rcx
   70b9f:	cmp    $0x38,%r15d
   70ba3:	jne    70bb0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0xc0>
   70ba5:	mov    -0x20(%rsp),%r8
   70baa:	jmp    70bf0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x100>
   70bac:	nopl   0x0(%rax)
   70bb0:	mov    -0x18(%rsp),%r14
   70bb5:	mov    -0x20(%rsp),%r8
   70bba:	nopw   0x0(%rax,%rax,1)
   70bc0:	mov    (%r8),%rbx
   70bc3:	add    $0x8,%r8
   70bc7:	lea    (%rbx,%rbx,4),%rbx
   70bcb:	shl    $0x8,%rbx
   70bcf:	inc    %r14
   70bd2:	vpmovsxbw (%rcx,%rbx,1),%zmm1
   70bd9:	vpaddw %zmm1,%zmm0,%zmm0
   70bdf:	jne    70bc0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0xd0>
   70be1:	cmpq   $0x38,-0x8(%rsp)
   70be7:	jb     70cc4 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x1d4>
   70bed:	nopl   (%rax)
   70bf0:	mov    (%r8),%rbx
   70bf3:	mov    0x8(%r8),%r14
   70bf7:	lea    (%rbx,%rbx,4),%rbx
   70bfb:	shl    $0x8,%rbx
   70bff:	vpmovsxbw (%rcx,%rbx,1),%zmm1
   70c06:	lea    (%r14,%r14,4),%rbx
   70c0a:	shl    $0x8,%rbx
   70c0e:	vpmovsxbw (%rcx,%rbx,1),%zmm3
   70c15:	mov    0x10(%r8),%rbx
   70c19:	lea    (%rbx,%rbx,4),%rbx
   70c1d:	shl    $0x8,%rbx
   70c21:	vpmovsxbw (%rcx,%rbx,1),%zmm2
   70c28:	mov    0x18(%r8),%rbx
   70c2c:	vpaddw %zmm1,%zmm0,%zmm0
   70c32:	lea    (%rbx,%rbx,4),%rbx
   70c36:	shl    $0x8,%rbx
   70c3a:	vpmovsxbw (%rcx,%rbx,1),%zmm4
   70c41:	mov    0x20(%r8),%rbx
   70c45:	vpaddw %zmm2,%zmm3,%zmm1
   70c4b:	vpaddw %zmm1,%zmm0,%zmm0
   70c51:	lea    (%rbx,%rbx,4),%rbx
   70c55:	shl    $0x8,%rbx
   70c59:	vpmovsxbw (%rcx,%rbx,1),%zmm2
   70c60:	mov    0x28(%r8),%rbx
   70c64:	lea    (%rbx,%rbx,4),%rbx
   70c68:	shl    $0x8,%rbx
   70c6c:	vpmovsxbw (%rcx,%rbx,1),%zmm3
   70c73:	mov    0x30(%r8),%rbx
   70c77:	vpaddw %zmm2,%zmm4,%zmm1
   70c7d:	lea    (%rbx,%rbx,4),%rbx
   70c81:	shl    $0x8,%rbx
   70c85:	vpaddw %zmm3,%zmm1,%zmm1
   70c8b:	vpmovsxbw (%rcx,%rbx,1),%zmm3
   70c92:	mov    0x38(%r8),%rbx
   70c96:	add    $0x40,%r8
   70c9a:	vpaddw %zmm1,%zmm0,%zmm0
   70ca0:	lea    (%rbx,%rbx,4),%rbx
   70ca4:	shl    $0x8,%rbx
   70ca8:	vpmovsxbw (%rcx,%rbx,1),%zmm2
   70caf:	vpaddw %zmm2,%zmm3,%zmm1
   70cb5:	vpaddw %zmm1,%zmm0,%zmm0
   70cbb:	cmp    %r11,%r8
   70cbe:	jne    70bf0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x100>
   70cc4:	cmp    $0x38,%r12d
   70cc8:	jne    70cd0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x1e0>
   70cca:	mov    %rax,%r8
   70ccd:	jmp    70d10 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x220>
   70ccf:	nop
   70cd0:	mov    %r9,%r14
   70cd3:	mov    %rax,%r8
   70cd6:	cs nopw 0x0(%rax,%rax,1)
   70ce0:	mov    (%r8),%rbx
   70ce3:	add    $0x8,%r8
   70ce7:	lea    (%rbx,%rbx,4),%rbx
   70ceb:	shl    $0x8,%rbx
   70cef:	inc    %r14
   70cf2:	vpmovsxbw (%rcx,%rbx,1),%zmm1
   70cf9:	vpsubw %zmm1,%zmm0,%zmm0
   70cff:	jne    70ce0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x1f0>
   70d01:	cmpq   $0x38,-0x10(%rsp)
   70d07:	jb     70b80 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x90>
   70d0d:	nopl   (%rax)
   70d10:	mov    (%r8),%rbx
   70d13:	mov    0x8(%r8),%r14
   70d17:	lea    (%rbx,%rbx,4),%rbx
   70d1b:	shl    $0x8,%rbx
   70d1f:	vpmovsxbw (%rcx,%rbx,1),%zmm1
   70d26:	lea    (%r14,%r14,4),%rbx
   70d2a:	shl    $0x8,%rbx
   70d2e:	vpmovsxbw (%rcx,%rbx,1),%zmm2
   70d35:	mov    0x10(%r8),%rbx
   70d39:	lea    (%rbx,%rbx,4),%rbx
   70d3d:	shl    $0x8,%rbx
   70d41:	vpmovsxbw (%rcx,%rbx,1),%zmm5
   70d48:	mov    0x18(%r8),%rbx
   70d4c:	vpaddw %zmm2,%zmm1,%zmm1
   70d52:	lea    (%rbx,%rbx,4),%rbx
   70d56:	shl    $0x8,%rbx
   70d5a:	vpmovsxbw (%rcx,%rbx,1),%zmm3
   70d61:	mov    0x20(%r8),%rbx
   70d65:	lea    (%rbx,%rbx,4),%rbx
   70d69:	shl    $0x8,%rbx
   70d6d:	vpmovsxbw (%rcx,%rbx,1),%zmm4
   70d74:	mov    0x28(%r8),%rbx
   70d78:	vpaddw %zmm3,%zmm5,%zmm2
   70d7e:	vpaddw %zmm2,%zmm1,%zmm1
   70d84:	lea    (%rbx,%rbx,4),%rbx
   70d88:	shl    $0x8,%rbx
   70d8c:	vpmovsxbw (%rcx,%rbx,1),%zmm3
   70d93:	mov    0x30(%r8),%rbx
   70d97:	lea    (%rbx,%rbx,4),%rbx
   70d9b:	shl    $0x8,%rbx
   70d9f:	vpaddw %zmm3,%zmm4,%zmm2
   70da5:	vpmovsxbw (%rcx,%rbx,1),%zmm4
   70dac:	mov    0x38(%r8),%rbx
   70db0:	add    $0x40,%r8
   70db4:	lea    (%rbx,%rbx,4),%rbx
   70db8:	shl    $0x8,%rbx
   70dbc:	vpmovsxbw (%rcx,%rbx,1),%zmm3
   70dc3:	vpaddw %zmm4,%zmm2,%zmm2
   70dc9:	vpaddw %zmm2,%zmm1,%zmm1
   70dcf:	vpaddw %zmm3,%zmm1,%zmm1
   70dd5:	vpsubw %zmm1,%zmm0,%zmm0
   70ddb:	cmp    %r10,%r8
   70dde:	jne    70d10 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x220>
   70de4:	jmp    70b80 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x90>
   70de9:	test   %rcx,%rcx
   70dec:	je     710e7 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x5f7>
   70df2:	mov    %r12,%r8
   70df5:	add    $0xfffffffffffffff8,%r8
   70df9:	mov    $0x28,%r9d
   70dff:	xor    %r11d,%r11d
   70e02:	mov    %r8d,%ecx
   70e05:	shr    $0x3,%ecx
   70e08:	mov    %r8,%r13
   70e0b:	and    $0x38,%r8d
   70e0f:	inc    %ecx
   70e11:	and    $0x7,%ecx
   70e14:	neg    %rcx
   70e17:	jmp    70e34 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x344>
   70e19:	nopl   0x0(%rax)
   70e20:	vmovdqu64 %zmm0,(%rdi,%r11,2)
   70e27:	add    $0x20,%r11
   70e2b:	dec    %r9
   70e2e:	je     710d9 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x5e9>
   70e34:	vmovdqu64 (%rsi,%r11,2),%zmm0
   70e3b:	lea    (%rdx,%r11,1),%rbx
   70e3f:	cmp    $0x38,%r8d
   70e43:	jne    70e50 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x360>
   70e45:	mov    %rax,%r14
   70e48:	jmp    70e90 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x3a0>
   70e4a:	nopw   0x0(%rax,%rax,1)
   70e50:	mov    %rcx,%r15
   70e53:	mov    %rax,%r14
   70e56:	cs nopw 0x0(%rax,%rax,1)
   70e60:	mov    (%r14),%r12
   70e63:	add    $0x8,%r14
   70e67:	lea    (%r12,%r12,4),%r12
   70e6b:	shl    $0x8,%r12
   70e6f:	inc    %r15
   70e72:	vpmovsxbw (%rbx,%r12,1),%zmm1
   70e79:	vpsubw %zmm1,%zmm0,%zmm0
   70e7f:	jne    70e60 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x370>
   70e81:	cmp    $0x38,%r13
   70e85:	jb     70e20 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x330>
   70e87:	nopw   0x0(%rax,%rax,1)
   70e90:	mov    (%r14),%r15
   70e93:	mov    0x8(%r14),%r12
   70e97:	lea    (%r15,%r15,4),%r15
   70e9b:	shl    $0x8,%r15
   70e9f:	vpmovsxbw (%rbx,%r15,1),%zmm1
   70ea6:	lea    (%r12,%r12,4),%r15
   70eaa:	shl    $0x8,%r15
   70eae:	vpmovsxbw (%rbx,%r15,1),%zmm2
   70eb5:	mov    0x10(%r14),%r15
   70eb9:	lea    (%r15,%r15,4),%r15
   70ebd:	shl    $0x8,%r15
   70ec1:	vpmovsxbw (%rbx,%r15,1),%zmm5
   70ec8:	mov    0x18(%r14),%r15
   70ecc:	vpaddw %zmm2,%zmm1,%zmm1
   70ed2:	lea    (%r15,%r15,4),%r15
   70ed6:	shl    $0x8,%r15
   70eda:	vpmovsxbw (%rbx,%r15,1),%zmm3
   70ee1:	mov    0x20(%r14),%r15
   70ee5:	lea    (%r15,%r15,4),%r15
   70ee9:	shl    $0x8,%r15
   70eed:	vpmovsxbw (%rbx,%r15,1),%zmm4
   70ef4:	mov    0x28(%r14),%r15
   70ef8:	vpaddw %zmm3,%zmm5,%zmm2
   70efe:	vpaddw %zmm2,%zmm1,%zmm1
   70f04:	lea    (%r15,%r15,4),%r15
   70f08:	shl    $0x8,%r15
   70f0c:	vpmovsxbw (%rbx,%r15,1),%zmm3
   70f13:	mov    0x30(%r14),%r15
   70f17:	lea    (%r15,%r15,4),%r15
   70f1b:	shl    $0x8,%r15
   70f1f:	vpaddw %zmm3,%zmm4,%zmm2
   70f25:	vpmovsxbw (%rbx,%r15,1),%zmm4
   70f2c:	mov    0x38(%r14),%r15
   70f30:	add    $0x40,%r14
   70f34:	lea    (%r15,%r15,4),%r15
   70f38:	shl    $0x8,%r15
   70f3c:	vpmovsxbw (%rbx,%r15,1),%zmm3
   70f43:	vpaddw %zmm4,%zmm2,%zmm2
   70f49:	vpaddw %zmm2,%zmm1,%zmm1
   70f4f:	vpaddw %zmm3,%zmm1,%zmm1
   70f55:	vpsubw %zmm1,%zmm0,%zmm0
   70f5b:	cmp    %r10,%r14
   70f5e:	jne    70e90 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x3a0>
   70f64:	jmp    70e20 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x330>
   70f69:	and    $0x7,%r8d
   70f6d:	mov    %r9d,%eax
   70f70:	and    $0x38,%eax
   70f73:	mov    $0x28,%ecx
   70f78:	mov    %r9,%rbx
   70f7b:	xor    %r10d,%r10d
   70f7e:	neg    %r8
   70f81:	mov    %r8,%r9
   70f84:	jmp    70fa4 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x4b4>
   70f86:	cs nopw 0x0(%rax,%rax,1)
   70f90:	vmovdqu64 %zmm0,(%rdi,%r10,2)
   70f97:	add    $0x20,%r10
   70f9b:	dec    %rcx
   70f9e:	je     710d9 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x5e9>
   70fa4:	vmovdqu64 (%rsi,%r10,2),%zmm0
   70fab:	lea    (%rdx,%r10,1),%r14
   70faf:	cmp    $0x38,%eax
   70fb2:	jne    70fc0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x4d0>
   70fb4:	mov    -0x20(%rsp),%r15
   70fb9:	jmp    71000 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x510>
   70fbb:	nopl   0x0(%rax,%rax,1)
   70fc0:	mov    -0x20(%rsp),%r15
   70fc5:	mov    %r9,%r8
   70fc8:	nopl   0x0(%rax,%rax,1)
   70fd0:	mov    (%r15),%r12
   70fd3:	add    $0x8,%r15
   70fd7:	lea    (%r12,%r12,4),%r12
   70fdb:	shl    $0x8,%r12
   70fdf:	inc    %r8
   70fe2:	vpmovsxbw (%r14,%r12,1),%zmm1
   70fe9:	vpaddw %zmm1,%zmm0,%zmm0
   70fef:	jne    70fd0 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x4e0>
   70ff1:	cmp    $0x38,%rbx
   70ff5:	jb     70f90 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x4a0>
   70ff7:	nopw   0x0(%rax,%rax,1)
   71000:	mov    (%r15),%r8
   71003:	mov    0x8(%r15),%r12
   71007:	lea    (%r8,%r8,4),%r8
   7100b:	shl    $0x8,%r8
   7100f:	vpmovsxbw (%r14,%r8,1),%zmm1
   71016:	lea    (%r12,%r12,4),%r8
   7101a:	shl    $0x8,%r8
   7101e:	vpmovsxbw (%r14,%r8,1),%zmm3
   71025:	mov    0x10(%r15),%r8
   71029:	lea    (%r8,%r8,4),%r8
   7102d:	shl    $0x8,%r8
   71031:	vpmovsxbw (%r14,%r8,1),%zmm2
   71038:	mov    0x18(%r15),%r8
   7103c:	vpaddw %zmm1,%zmm0,%zmm0
   71042:	lea    (%r8,%r8,4),%r8
   71046:	shl    $0x8,%r8
   7104a:	vpmovsxbw (%r14,%r8,1),%zmm4
   71051:	mov    0x20(%r15),%r8
   71055:	vpaddw %zmm2,%zmm3,%zmm1
   7105b:	vpaddw %zmm1,%zmm0,%zmm0
   71061:	lea    (%r8,%r8,4),%r8
   71065:	shl    $0x8,%r8
   71069:	vpmovsxbw (%r14,%r8,1),%zmm2
   71070:	mov    0x28(%r15),%r8
   71074:	lea    (%r8,%r8,4),%r8
   71078:	shl    $0x8,%r8
   7107c:	vpmovsxbw (%r14,%r8,1),%zmm3
   71083:	mov    0x30(%r15),%r8
   71087:	vpaddw %zmm2,%zmm4,%zmm1
   7108d:	lea    (%r8,%r8,4),%r8
   71091:	shl    $0x8,%r8
   71095:	vpaddw %zmm3,%zmm1,%zmm1
   7109b:	vpmovsxbw (%r14,%r8,1),%zmm3
   710a2:	mov    0x38(%r15),%r8
   710a6:	add    $0x40,%r15
   710aa:	vpaddw %zmm1,%zmm0,%zmm0
   710b0:	lea    (%r8,%r8,4),%r8
   710b4:	shl    $0x8,%r8
   710b8:	vpmovsxbw (%r14,%r8,1),%zmm2
   710bf:	vpaddw %zmm2,%zmm3,%zmm1
   710c5:	vpaddw %zmm1,%zmm0,%zmm0
   710cb:	cmp    %r11,%r15
   710ce:	jne    71000 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x510>
   710d4:	jmp    70f90 <_RNvNtNtNtNtCsio7Oyl2UYcL_11chessengine4eval4nnue9sota_simd6avx51214i8_deltas_into+0x4a0>
   710d9:	pop    %rbx
   710da:	pop    %r12
   710dc:	pop    %r13
   710de:	pop    %r14
   710e0:	pop    %r15
   710e2:	pop    %rbp
   710e3:	vzeroupper
   710e6:	ret
   710e7:	mov    $0xa00,%edx
   710ec:	pop    %rbx
   710ed:	pop    %r12
   710ef:	pop    %r13
   710f1:	pop    %r14
   710f3:	pop    %r15
   710f5:	pop    %rbp
   710f6:	jmp    *0x6b7fc(%rip)        # dc8f8 <memcpy@GLIBC_2.14>
