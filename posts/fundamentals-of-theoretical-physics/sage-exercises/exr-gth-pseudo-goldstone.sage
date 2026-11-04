# Exercise exr-gth-pseudo-goldstone: a small explicit breaking -eps u of the U(1) of V = (lam/4)(u^2 + v^2 - mu^2)^2
# (u, v real fields with kinetic terms (du)^2/2 + (dv)^2/2) picks the vacuum
# (u0, 0) with lam (u0^2 - mu^2) u0 = eps and gives the would-be Goldstone mode v the mass m^2 = eps/u0.
u, v = var("u v", domain="real"); lam, mu, eps = 1, 1, 1/20
V = lam / 4 * (u^2 + v^2 - mu^2)^2 - eps * u
u0 = max(diff(V, u).subs(v=0).roots(ring=RR, multiplicities=False))
H = matrix(RR, 2, 2, lambda i, j: diff(V, [u, v][i], [u, v][j]).subs(u=u0, v=0))
print("vacuum u0 =", round(u0, 8))
print("Hessian at the vacuum:"); print(H)
print("[PASS] the Goldstone direction v acquires m^2 = eps/u0:", abs(H[1, 1] - eps / u0) < 1e-12 and abs(H[0, 1]) < 1e-12)
print("[PASS] the radial mass stays close to the unbroken value 2 lam mu^2:", abs(H[0, 0] - 2 * lam * mu^2) < 4 * eps)
L, M, E, U = var("L M E U", domain="positive")
print("[PASS] symbolically, d^2V/dv^2 at (U, 0) is lam (U^2 - mu^2), which equals eps/U on the vacuum condition:", bool((diff(L / 4 * (u^2 + v^2 - M^2)^2 - E * u, v, 2).subs(u=U, v=0) - L * (U^2 - M^2)) == 0))
