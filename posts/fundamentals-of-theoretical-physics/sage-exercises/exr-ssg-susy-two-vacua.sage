# Exercise exr-ssg-susy-two-vacua (integrative): the Wess-Zumino model with W = lambda (Phi^3/3 - mu^2 Phi), lambda, mu > 0.
# F-term, potential, vacua, the mass spectrum in each vacuum (scalar Hessian against fermion mass), the supertrace, the
# Witten index as the number of supersymmetric vacua, and the tension of the BPS domain wall 2|W(mu) - W(-mu)|.
lam, mu = var("lam mu", domain="positive"); p, c = var("p c")          # phi and phi^* as independent symbols
W = lam * (p^3 / 3 - mu^2 * p); Wp = diff(W, p); Wpp = diff(W, p, 2)
V = Wp * Wp.subs(p=c)
vac = [s.rhs() for s in solve(Wp == 0, p)]
print("W' =", Wp.factor(), "; vacua:", vac)
print("[PASS] two supersymmetric vacua phi = -mu, +mu with V = 0:", set(vac) == {-mu, mu} and all(bool(V.subs(p=v, c=v) == 0) for v in vac))
x, w = var("x w", domain="real")
Vr = V.subs(p=(x + I * w) / sqrt(2), c=(x - I * w) / sqrt(2)).expand()
for v in vac:
    H = matrix(SR, 2, 2, lambda i, j: diff(Vr, (x, w)[i], (x, w)[j])).subs(x=sqrt(2) * v, w=0).simplify_full()
    mf2 = (Wpp.subs(p=v))^2
    print("vacuum", v, ": scalar mass^2 eigenvalues", H.eigenvalues(), "; fermion mass^2", mf2)
    print("[PASS] at phi = %s both real scalars and the Weyl fermion have mass^2 4 lambda^2 mu^2; supertrace 2*4l^2mu^2 - 2*4l^2mu^2 = 0:" % v,
          H == 4 * lam^2 * mu^2 * identity_matrix(2) and bool(mf2 == 4 * lam^2 * mu^2))
print("[PASS] Witten index = number of massive supersymmetric vacua = 2 (each contributes one bosonic ground state):", len(vac) == 2)
T = 2 * abs(W.subs(p=mu) - W.subs(p=-mu))
print("BPS wall tension 2|Delta W| =", T.simplify_full())
print("[PASS] the tension is 8 lambda mu^3 / 3:", bool((T - 8 * lam * mu^3 / 3).simplify_full() == 0))
xx = var("xx"); kink = -mu * tanh(lam * mu * xx)                       # solves phi' = W'(phi): -lam mu^2 sech^2 = lam(mu^2 tanh^2 - mu^2)
print("[PASS] phi(x) = -mu tanh(lambda mu x) solves phi' = W'(phi) and runs from mu to -mu:",
      bool((diff(kink, xx) - Wp.subs(p=kink)).simplify_full() == 0) and bool(limit(kink, xx=-oo) == mu) and bool(limit(kink, xx=oo) == -mu))
