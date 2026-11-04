# Exercise exr-lrep-dj-harmonics: the middle column of the spin-l rotation matrix is a list of spherical harmonics,
# sqrt((2l+1)/(4 pi)) conj(D^l_{m0}(phi, theta, psi)) = Y_lm(theta, phi - pi/2), the harmonics at the direction R e_3,
# whose spherical angles are (theta, phi - pi/2). Sage's spherical_harmonic uses the Condon-Shortley phase.
th, ph, ps = var("th ph ps", domain="real")
def spin_matrices(j):
    ms = [j - k for k in range(2 * j + 1)]; n = len(ms)
    J3 = diagonal_matrix(SR, ms)
    Jp = matrix(SR, n, n, lambda r, c: sqrt(j * (j + 1) - ms[c] * (ms[c] + 1)) if r == c - 1 else 0)
    return ms, J3, Jp, Jp.transpose()
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
def rot(a, u): return identity_matrix(SR, 3) + sin(u) * L[a] + (1 - cos(u)) * L[a]^2
Re3 = (rot(2, ph) * rot(0, th) * rot(2, ps)).column(2).apply_map(lambda e: e.simplify_full())
print("R e_3 =", Re3, "= (sin th cos(ph - pi/2), sin th sin(ph - pi/2), cos th):",
      bool((Re3 - vector([sin(th) * cos(ph - pi / 2), sin(th) * sin(ph - pi / 2), cos(th)])).apply_map(lambda e: e.simplify_full()) == 0))
for l in [1, 2, 3]:
    ms, J3, Jp, Jm = spin_matrices(l)
    D = (-I * ph * J3).exp() * (-I * th * (Jp + Jm) / 2).exp() * (-I * ps * J3).exp()
    col = D.column(l).apply_map(lambda e: e.rectform().simplify_full())      # the column m' = 0
    ok = []
    for k, m in enumerate(ms):
        lhs = sqrt((2 * l + 1) / (4 * pi)) * col[k].conjugate()
        rhs = spherical_harmonic(l, m, th, ph - pi / 2)
        ok.append(bool((lhs - rhs).rectform().simplify_full().canonicalize_radical() == 0))
    print(f"l = {l}: column m' = 0 independent of psi: {all(bool(diff(c, ps) == 0) for c in col)}; equals Y_lm(theta, phi - pi/2) for m = {ms}: {ok}")
