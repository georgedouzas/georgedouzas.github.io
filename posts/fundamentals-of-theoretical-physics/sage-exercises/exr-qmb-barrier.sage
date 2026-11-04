# Exercise exr-qmb-barrier (Cini et al., Sec. 13.2): transmission through a rectangular barrier of height V0 and width L
# for E < V0, T = [1 + V0^2 sinh^2(kappa L)/(4 E (V0 - E))]^(-1), checked against the matching conditions solved by
# Sage, and evaluated for an electron with E = 1 eV, V0 = 5 eV, L = 0.5 nm (hbar^2/2m = 0.0381 eV nm^2).
k, q, L, x = var('k q L x'); assume(k > 0, q > 0, L > 0)
A, B, C, F, G = var('A B C F G')   # psi = e^{ikx} + A e^{-ikx}; B e^{qx} + C e^{-qx}; F e^{ikx}
eqs = [1 + A == B + C, I * k * (1 - A) == q * (B - C),
       B * exp(q * L) + C * exp(-q * L) == F * exp(I * k * L), q * (B * exp(q * L) - C * exp(-q * L)) == I * k * F * exp(I * k * L)]
sol = solve(eqs, A, B, C, F, solution_dict=True)[0]
Tm = (sol[F] * sol[F].conjugate()).simplify_full()
E, V0 = k^2, k^2 + q^2                 # units with hbar^2/2m = 1
formula = 1 / (1 + V0^2 * sinh(q * L)^2 / (4 * E * (V0 - E)))
print("|F|^2 equals the closed form (checked at three points):",
      all(bool(abs((Tm - formula).subs(k=kv, q=qv, L=Lv).n()) < 1e-12) for kv, qv, Lv in [(1, 2, 1), (0.5, 1.5, 2), (2, 0.7, 0.3)]))
c = 0.0381
for Lnm in (0.3, 0.5, 0.7):
    Ev, Vv = 1.0, 5.0
    kap = sqrt((Vv - Ev) / c)
    Tex = 1 / (1 + Vv^2 * sinh(kap * Lnm)^2 / (4 * Ev * (Vv - Ev)))
    Tap = 16 * Ev * (Vv - Ev) / Vv^2 * exp(-2 * kap * Lnm)
    print("L = %.1f nm: kappa L = %.3f, T = %.3e, thick-barrier estimate %.3e" % (Lnm, kap * Lnm, Tex, Tap))
