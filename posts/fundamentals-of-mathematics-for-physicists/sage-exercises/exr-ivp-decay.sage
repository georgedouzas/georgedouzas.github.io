# Exercise exr-ivp-decay: y' = -2y with y(0) = 5 by desolve, and the half-life.
x = var("x")
y = function("y")(x)
sol = desolve(diff(y, x) == -2 * y, y, ics=[0, 5])
print("solution:", sol)
print("check y' + 2y = 0 and y(0) = 5:", bool((diff(sol, x) + 2 * sol).simplify_full() == 0), bool(sol.subs(x=0) == 5))
half = log(2) / 2
print("y(log(2)/2) =", sol.subs(x=half).simplify_full(), "so the half-life is log(2)/2 =", n(half), "| exact:", bool(sol.subs(x=half) == 5 / 2))
