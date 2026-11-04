# Exercise exr-ssg-aux-classical: an auxiliary field in classical mechanics. L = xdot^2/2 + F^2/2 + F W'(x) has the
# algebraic equation F = -W'(x); substituting gives L = xdot^2/2 - W'^2/2, and the Euler-Lagrange equation of x
# from the original L, with F eliminated afterwards, is the same as from the reduced one: xddot = -W' W''.
# Position X, velocity Xd and F are treated as independent symbols, Wp stands for W'.
X, Xd, F = var("X Xd F"); Wp = function("Wp")
L = Xd^2 / 2 + F^2 / 2 + F * Wp(X)
Fsol = solve(diff(L, F) == 0, F)[0].rhs()
print("dL/dF =", diff(L, F), "=> F =", Fsol, "(no time derivative of F: the equation is algebraic)")
Lred = L.subs(F == Fsol).expand()
print("[PASS] substituting F = -W'(x) gives L = xdot^2/2 - W'(x)^2/2:", bool((Lred - (Xd^2 / 2 - Wp(X)^2 / 2)).expand() == 0))
xdd_full = diff(L, X).subs(F == Fsol)                 # xddot = dL/dx, then eliminate F
xdd_red = diff(Lred, X)
print("[PASS] the equation of motion xddot = F W'' with F = -W' equals that of the reduced Lagrangian, xddot = -W' W'':", bool((xdd_full - xdd_red).expand() == 0))
q = var("q")
print("[PASS] for W = x^3/3 - x the potential (x^2 - 1)^2/2 vanishes exactly at the critical points x = -1, 1 of W:",
      set(s.rhs() for s in solve(q^2 - 1 == 0, q)) == {-1, 1})
