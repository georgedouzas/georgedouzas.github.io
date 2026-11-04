# Exercise exr-ls-desolve-system: the coupled tanks x1' = -x1 + x2, x2' = x1 - x2 with x(0) = (3, 1) by desolve_system;
# the total is conserved and the difference decays like e^(-2t).
t = var("t")
x1 = function("x1")(t); x2 = function("x2")(t)
sol = desolve_system([diff(x1, t) == -x1 + x2, diff(x2, t) == x1 - x2], [x1, x2], ics=[0, 3, 1])
print("solution:", sol)
s1, s2 = sol[0].rhs(), sol[1].rhs()
print("x1 + x2 =", (s1 + s2).simplify_full(), "| x1 - x2 =", (s1 - s2).simplify_full())
print("both equations satisfied:", bool((diff(s1, t) + s1 - s2).simplify_full() == 0) and bool((diff(s2, t) - s1 + s2).simplify_full() == 0))
