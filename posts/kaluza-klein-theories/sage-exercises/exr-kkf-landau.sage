# Exercise exr-kkf-landau: Landau and Zeeman levels E - m = omega (N + 1/2) - g omega s / 2 of a spin-1/2 particle in the
# Pauli approximation, for g = 1 (Kaluza-Klein modes) and g = 2 (Dirac particles), in units of omega = |q| B / m.
from collections import Counter
for gg in (1, 2):
    lev = Counter()
    for N in range(8):
        for s in (QQ(1) / 2, -QQ(1) / 2):
            lev[N + QQ(1) / 2 - gg * s / 2] += 1
    low = sorted(lev)[:5]
    print("g = %d: five lowest levels (E - m)/omega with degeneracy:" % gg, [(e, lev[e]) for e in low])
print("g = 2 ground level at E = m:", min(N + QQ(1) / 2 - 2 * s / 2 for N in range(3) for s in (QQ(1) / 2, -QQ(1) / 2)) == 0)
print("g = 1 levels all non-degenerate:", all(v == 1 for v in Counter(N + QQ(1) / 2 - s / 2 for N in range(8) for s in (QQ(1) / 2, -QQ(1) / 2)).values()))
