# Exercise exr-gth-tilted-double-well: the real scalar with V = (lam/4)(phi^2 - mu^2)^2 - eps phi, lam = mu = 1,
# eps = 1/10. The small tilt lifts the degeneracy of the two vacua: one true vacuum near mu + eps/(2 lam mu^2), one
# false (metastable) vacuum near -mu, and a maximum near 0.
phi = var("phi"); lam, mu, eps = 1, 1, 1/10
V = lam / 4 * (phi^2 - mu^2)^2 - eps * phi
crit = sorted([r for r in diff(V, phi).roots(ring=RR, multiplicities=False)])
print("critical points:", [round(c, 6) for c in crit])
for c in crit:
    print("  phi = %.6f: V = %.6f, V'' = %.6f" % (c, V.subs(phi=c), diff(V, phi, 2).subs(phi=c)))
lo, mid, hi = crit
print("[PASS] the right minimum is the true vacuum, the left one is a false vacuum, the middle point a maximum:", bool(V.subs(phi=hi) < V.subs(phi=lo)) and bool(diff(V, phi, 2).subs(phi=mid) < 0) and bool(diff(V, phi, 2).subs(phi=lo) > 0) and bool(diff(V, phi, 2).subs(phi=hi) > 0))
print("first-order estimate mu + eps/(2 lam mu^2) =", float(mu + eps / (2 * lam * mu^2)), "; exact:", round(hi, 6))
print("[PASS] the estimate is good to O(eps^2):", abs(hi - float(mu + eps / (2 * lam * mu^2))) < eps^2)
print("energy splitting V(false) - V(true) =", round(V.subs(phi=lo) - V.subs(phi=hi), 6), "~ 2 eps mu =", float(2 * eps * mu))
