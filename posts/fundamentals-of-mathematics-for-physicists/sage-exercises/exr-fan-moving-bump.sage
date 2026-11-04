# Exercise exr-fan-moving-bump: f_n(x) = n x exp(1 - n x) on [0, 1]: maximum 1 at x = 1/n, pointwise limit 0, L^1 norm -> 0.
x = var('x'); n = var('n'); assume(n > 1)
f = n * x * exp(1 - n * x)
print("critical point:", solve(diff(f, x) == 0, x), " value there:", f.subs(x=1 / n))
print("L^1 norm:", integrate(f, x, 0, 1).simplify_full())
for nv in (1, 10, 100, 1000):
    print("n = %4d: sup = %.4f, L^1 = %.5f, f_n(0.3) = %.3e" % (nv, find_local_maximum(f.subs(n=nv), 0, 1)[0], integrate(f.subs(n=nv), x, 0, 1).n(), f.subs(n=nv, x=0.3).n()))
