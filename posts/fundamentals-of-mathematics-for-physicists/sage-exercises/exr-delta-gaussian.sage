# Exercise exr-delta-gaussian: the Gaussian sequence g_eps = exp(-x^2/eps^2)/(sqrt(pi) eps) integrated against cos x,
# x^2 and e^x, and the limit eps -> 0, which gives the value of the test function at 0.
x, eps = var("x eps")
assume(eps > 0)
g = exp(-x ** 2 / eps ** 2) / (sqrt(pi) * eps)
print("integral of g_eps:", integrate(g, x, -oo, oo).simplify_full())
for f in [cos(x), x ** 2, exp(x)]:
    I = integrate(g * f, x, -oo, oo).simplify_full()
    print("f =", f, ": integral =", I, "| limit eps -> 0:", limit(I, eps=0), "| f(0) =", f.subs(x=0))
