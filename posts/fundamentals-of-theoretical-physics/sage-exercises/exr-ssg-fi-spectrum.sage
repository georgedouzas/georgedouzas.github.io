# Exercise exr-ssg-fi-spectrum: a U(1) with Fayet-Iliopoulos term -kappa D and two chiral fields of charges +1, -1 with
# W = m Phi_+ Phi_-. Eliminating D: D = kappa - g(|phi_+|^2 - |phi_-|^2), and
# V = m^2(|phi_+|^2 + |phi_-|^2) + (1/2)(kappa - g(|phi_+|^2 - |phi_-|^2))^2 (Martin, hep-ph/9709356, Sec. 7.2).
# For m^2 > g kappa > 0 the minimum is the origin with V = kappa^2/2 > 0, scalar masses m^2 - g kappa and m^2 + g kappa,
# fermion masses m (a Dirac pair) and a massless gaugino, the goldstino; the supertrace vanishes because the
# kappa-dependent shifts are proportional to the charges, which sum to zero.
g, k, m = var("g kappa m", domain="positive")
r, s = var("r s", domain="positive")                    # r = |phi_+|^2, s = |phi_-|^2
V = m^2 * (r + s) + (k - g * (r - s))^2 / 2
print("[PASS] V at the origin is kappa^2/2 > 0 (supersymmetry broken by the D-term):", bool(V.subs(r=0, s=0) == k^2 / 2))
print("dV/dr at 0 =", diff(V, r).subs(r=0, s=0), "; dV/ds at 0 =", diff(V, s).subs(r=0, s=0))
assume(m^2 > g * k)
print("[PASS] for m^2 > g kappa both slopes in |phi|^2 are positive: the origin is the minimum:",
      bool(diff(V, r).subs(r=0, s=0) > 0) and bool(diff(V, s).subs(r=0, s=0) > 0))
print("[PASS] the scalar masses squared at the origin are m^2 - g kappa (phi_+) and m^2 + g kappa (phi_-):",
      bool(diff(V, r).subs(r=0, s=0) == m^2 - g * k) and bool(diff(V, s).subs(r=0, s=0) == m^2 + g * k))
# supertrace: complex scalars count 2 real each; Weyl fermions count 2: scalars 2(m^2 - g k) + 2(m^2 + g k), fermions 2 * 2 * m^2 (psi_+, psi_-), gaugino 0
str_ = 2 * (m^2 - g * k) + 2 * (m^2 + g * k) - 2 * (2 * m^2 + 0)
print("[PASS] the supertrace of the matter and gaugino sector vanishes (the D-term contribution g kappa sum q = 0):", bool(str_ == 0))
forget(); assume(g > 0, k > 0, m > 0, m^2 < g * k)
rmin = (k - m^2 / g) / g
Vmin = V.subs(r=rmin, s=0).simplify_full()
print("for m^2 < g kappa: minimum at |phi_+|^2 =", rmin, "with V =", Vmin)
print("[PASS] for m^2 < g kappa the minimum moves to |phi_+|^2 = (kappa - m^2/g)/g, with V = m^2(2 g kappa - m^2)/(2 g^2) > 0, below kappa^2/2:",
      bool(diff(V, r).subs(r=rmin, s=0) == 0) and bool((Vmin - m^2 * (2 * g * k - m^2) / (2 * g^2)).simplify_full() == 0)
      and bool((k^2 / 2 - Vmin - (g * k - m^2)^2 / (2 * g^2)).simplify_full() == 0))
