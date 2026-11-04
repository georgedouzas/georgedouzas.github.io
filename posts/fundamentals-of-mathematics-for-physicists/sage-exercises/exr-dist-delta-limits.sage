# Exercise exr-dist-delta-limits: the Gaussian delta sequence against x^2 + 1, e^x and cos 2x, exactly in epsilon, and
# the Fourier kernel sin(Lx)/(pi x) against cos 2x as a function of L.
x, eps, L = var("x epsilon L")
assume(eps > 0)
g = exp(-x ** 2 / eps ** 2) / (eps * sqrt(pi))
for name, phi in [("x^2 + 1", x ** 2 + 1), ("e^x", exp(x)), ("cos 2x", cos(2 * x))]:
    v = integrate(g * phi, x, -oo, oo).simplify_full()
    print("int g_eps *", name, "=", v, "; limit eps -> 0:", limit(v, epsilon=0))
assume(L > 2)
print("int delta_L cos 2x for L > 2:", integrate(sin(L * x) / (pi * x) * cos(2 * x), x, -oo, oo))
forget(L > 2); assume(L > 0); assume(L < 2)
print("int delta_L cos 2x for 0 < L < 2:", integrate(sin(L * x) / (pi * x) * cos(2 * x), x, -oo, oo))
print("int delta_L cos 2x at L = 2:", integrate(sin(2 * x) / (pi * x) * cos(2 * x), x, -oo, oo))
