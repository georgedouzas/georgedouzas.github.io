# Exercise exr-ivp-second-order: q'' + 4q = 0 with q(0) = 1, q'(0) = 2 by the Laplace transform solver, then checked.
t = var("t")
q = function("q")(t)
sol = desolve_laplace(diff(q, t, 2) + 4 * q == 0, q, ics=[0, 1, 2])
print("solution q(t) =", sol)
print("q'' + 4q = 0:", bool((diff(sol, t, 2) + 4 * sol).simplify_full() == 0))
print("q(0) = 1 and q'(0) = 2:", bool(sol.subs(t=0) == 1), bool(diff(sol, t).subs(t=0) == 2))
