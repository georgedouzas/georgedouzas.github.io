# Exercise exr-ssg-witten-index: the Witten index Tr (-1)^F exp(-beta H) of the supersymmetric oscillator. Bosonic
# states |n, 0> have E = n omega, fermionic |n, 1> have E = (n + 1) omega; the two geometric series cancel except for
# the ground state, so the index is 1 for every beta, and the partition function is (1 + q)/(1 - q) with q = exp(-beta omega).
beta, w = var("beta omega", domain="positive"); n = var("n"); q = exp(-beta * w)
ZB = sum(q^n, n, 0, oo); ZF = sum(q^(n + 1), n, 0, oo)
print("Z_bosons =", ZB.simplify_full(), "  Z_fermions =", ZF.simplify_full())
print("[PASS] Tr (-1)^F e^{-beta H} = Z_B - Z_F = 1, independent of beta:", bool((ZB - ZF).simplify_full() == 1))
print("[PASS] Tr e^{-beta H} = (1 + q)/(1 - q):", bool((ZB + ZF - (1 + q) / (1 - q)).simplify_full() == 0))
# a truncated check with explicit matrices at beta*omega = 1
Nt = 40
ev = [(k, 0) for k in range(Nt)] + [(k + 1, 1) for k in range(Nt)]
R80 = RealField(80); idx = sum((-1)^f * R80(-k).exp() for k, f in ev)
print("truncated index at beta omega = 1:", idx)
print("[PASS] the truncated index equals 1 - e^{-40}, the unpaired top fermion of the cut-off:", abs(idx - 1 + R80(-40).exp()) < 1e-20)
