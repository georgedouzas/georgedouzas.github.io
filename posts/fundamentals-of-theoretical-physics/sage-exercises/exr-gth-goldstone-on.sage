# Exercise exr-gth-goldstone-on: the O(5) model V = (lam/4)(phi.phi - a^2)^2 at the vacuum a n, n = (1, 2, 0, -1, 3)/sqrt(15).
# The Hessian is 2 lam a^2 n n^T: four zero modes (the Goldstone bosons, dim O(5)/O(4) = 10 - 6 = 4) and one massive
# mode with m^2 = 2 lam a^2 along n.
lam, a = var("lam a"); assume(lam > 0, a > 0)
p = vector(SR, var("p1 p2 p3 p4 p5"))
V = lam / 4 * (p * p - a^2)^2
n = vector(SR, [1, 2, 0, -1, 3]) / sqrt(15)
vac = dict(zip(p, a * n))
grad = vector(SR, [diff(V, pi) for pi in p]).subs(vac)
H = matrix(SR, 5, 5, lambda i, j: diff(V, p[i], p[j])).subs(vac).simplify_full()
print("[PASS] a n is a critical point with V = 0:", grad.is_zero() and bool(V.subs(vac) == 0))
print("[PASS] the Hessian is 2 lam a^2 n n^T:", (H - 2 * lam * a^2 * n.column() * n.row()).simplify_full().is_zero())
ev = H.eigenvalues()
print("eigenvalues:", ev)
print("[PASS] four zero eigenvalues (Goldstone bosons) and one 2 lam a^2:", sorted([bool(e == 0) for e in ev]).count(True) == 4 and any(bool(e == 2 * lam * a^2) for e in ev))
print("dim O(5) - dim O(4) =", 5 * 4 // 2 - 4 * 3 // 2)
