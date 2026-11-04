# Exercise exr-dist-derivatives: the derivatives of the Gaussian delta sequence against test functions, exactly in
# epsilon, with the limits -phi'(0) and phi''(0).
x, eps = var("x epsilon")
assume(eps > 0)
g = exp(-x ** 2 / eps ** 2) / (eps * sqrt(pi))
for name, phi in [("x", x), ("x^2", x ** 2), ("e^x", exp(x))]:
    v = integrate(diff(g, x) * phi, x, -oo, oo).simplify_full()
    print("int g_eps' *", name, "=", v, "; limit:", limit(v, epsilon=0), "; -phi'(0) =", -diff(phi, x).subs(x=0))
for name, phi in [("x^2", x ** 2), ("cos x", cos(x))]:
    v = integrate(diff(g, x, 2) * phi, x, -oo, oo).simplify_full()
    print("int g_eps'' *", name, "=", v, "; limit:", limit(v, epsilon=0), "; phi''(0) =", diff(phi, x, 2).subs(x=0))
