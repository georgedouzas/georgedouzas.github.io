# Exercise exr-ssg-ladders: massless multiplets of N-extended supersymmetry from a Clifford vacuum of helicity lambda_0.
# The N creation operators raise the helicity by 1/2 each; the level-k states number binomial(N, k). We list the
# gravity multiplets (top helicity 2) for N = 1..8 with their CPT completion, count bosons and fermions, and check that
# the CPT-completed N = 7 multiplet coincides with N = 8.
def single(N, top):
    lam0 = top - QQ(N) / 2
    return {lam0 + QQ(k) / 2: binomial(N, k) for k in range(N + 1)}
def cpt(N, top):
    s = single(N, top); out = dict(s)
    if min(s) != -top:
        for h, n in s.items(): out[-h] = out.get(-h, 0) + n
    return out
for N in range(1, 9):
    st = cpt(N, 2)
    nb = sum(n for h, n in st.items() if h.denominator() == 1); nf = sum(n for h, n in st.items() if h.denominator() == 2)
    print("N = %d:" % N, ", ".join("%s:%d" % (h, st[h]) for h in sorted(st, reverse=True)), " bosons", nb, "fermions", nf)
print("[PASS] every multiplet has as many bosonic as fermionic states:", all(sum(n for h, n in cpt(N, 2).items() if h.denominator() == 1) == sum(n for h, n in cpt(N, 2).items() if h.denominator() == 2) for N in range(1, 9)))
print("[PASS] the CPT-completed N = 7 gravity multiplet equals the N = 8 one:", cpt(7, 2) == cpt(8, 2))
print("[PASS] the N = 8 multiplet has 256 states with helicity counts 1, 8, 28, 56, 70, 56, 28, 8, 1:",
      [cpt(8, 2)[2 - QQ(k) / 2] for k in range(9)] == [1, 8, 28, 56, 70, 56, 28, 8, 1] and sum(cpt(8, 2).values()) == 256)
print("[PASS] for N = 9 any ladder spans 9/2 units, so some |helicity| > 2:", all(max(abs(h) for h in single(9, t)) > 2 for t in [QQ(k) / 2 for k in range(-4, 10)]))
