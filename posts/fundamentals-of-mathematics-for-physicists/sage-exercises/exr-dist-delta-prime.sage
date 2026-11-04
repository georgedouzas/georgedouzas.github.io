# Exercise exr-dist-delta-prime: two sequences converging to delta', the derivative g_eps' and the dipole
# [g_eps(x + eps) - g_eps(x - eps)]/(2 eps), against x, x^3 and sin x, exactly in epsilon.
x, eps = var("x epsilon")
assume(eps > 0)
g = exp(-x ** 2 / eps ** 2) / (eps * sqrt(pi))
for name, phi in [("x", x), ("x^3", x ** 3), ("sin x", sin(x))]:
    v1 = integrate(diff(g, x) * phi, x, -oo, oo).simplify_full()
    v2 = ((integrate(g.subs(x=x + eps) * phi, x, -oo, oo) - integrate(g.subs(x=x - eps) * phi, x, -oo, oo)) / (2 * eps)).simplify_full()
    print(name, ": int g_eps' phi =", v1, "-> ", limit(v1, epsilon=0), "; int d_eps phi =", v2, "->", limit(v2, epsilon=0), "; -phi'(0) =", -diff(phi, x).subs(x=0))
