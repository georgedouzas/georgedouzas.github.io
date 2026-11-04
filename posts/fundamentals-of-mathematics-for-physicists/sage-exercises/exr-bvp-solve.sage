# Exercise exr-bvp-solve: -u'' = sin(pi x) on [0, 1] with u(0) = u(1) = 0 by desolve with two-point conditions, and the
# check against u = sin(pi x)/pi^2.
x = var("x")
u = function("u")(x)
sol = desolve(-diff(u, x, 2) == sin(pi * x), u, ics=[0, 0, 1, 0])
print("desolve with ics=[0, 0, 1, 0]:", sol)
print("equals sin(pi x)/pi^2:", bool((sol - sin(pi * x) / pi ** 2).simplify_full() == 0))
print("-u'' = sin(pi x), u(0) = 0, u(1) = 0:", bool((-diff(sol, x, 2) - sin(pi * x)).simplify_full() == 0), bool(sol.subs(x=0) == 0), bool(sol.subs(x=1).simplify_full() == 0))
