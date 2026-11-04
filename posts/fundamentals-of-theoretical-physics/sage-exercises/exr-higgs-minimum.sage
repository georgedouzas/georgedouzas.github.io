# Exercise exr-higgs-minimum: minimise V = -m^2 |phi|^2 + lambda |phi|^4 and relate it to the completed-square form.
r, m, lam = var('r m lam', domain='real'); assume(m > 0, lam > 0)
V = -m^2*r^2 + lam*r^4
crit = solve(diff(V, r) == 0, r); print("critical points:", crit)
r0 = sqrt(m^2/(2*lam)); print("minimum at r0 = m/sqrt(2 lam); V(r0) =", V.subs(r=r0).simplify_full(), "; second derivative:", diff(V, r, 2).subs(r=r0).simplify_full())
mu2 = m^2/(2*lam); print("completed square: V = lam (r^2 - mu^2)^2 - lam mu^4 with mu^2 = m^2/(2 lam):", bool((V - (lam*(r^2 - mu2)^2 - lam*mu2^2)).expand() == 0))
