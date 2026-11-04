# Exercise exr-dist-ft-constant: the transforms of e^{-eps x^2} and e^{-eps|x|} tend to 2 pi delta: integrated against
# two test functions, the limits eps -> 0 are 2 pi phi(0).
x, k, eps = var("x k epsilon")
assume(eps > 0); assume(k, "real")
Fg = integrate(exp(-eps * x ** 2) * exp(-I * k * x), x, -oo, oo).simplify_full()
Fe = integrate(exp(-eps * abs(x)) * exp(-I * k * x), x, -oo, oo).simplify_full()
print("transform of e^{-eps x^2}:", Fg, "; of e^{-eps|x|}:", Fe)
print("both have integral 2 pi:", bool(integrate(Fg, k, -oo, oo) == 2 * pi) and bool(integrate(Fe, k, -oo, oo) == 2 * pi))
for name, phi in [("e^{-k^2/2}", exp(-k ** 2 / 2)), ("cos k", cos(k))]:
    vg = integrate(Fg * phi, k, -oo, oo).canonicalize_radical()
    print("Gaussian damping against", name, ":", vg, "->", limit(vg, epsilon=0), "; 2 pi phi(0) =", 2 * pi * phi.subs(k=0))
for name, phi in [("cos k", cos(k)), ("1/(1 + k^2)", 1 / (1 + k ** 2))]:
    ve = integrate(Fe * phi, k, -oo, oo).simplify_full()
    print("exponential damping against", name, ":", ve, "->", limit(ve, epsilon=0), "; 2 pi phi(0) =", 2 * pi * phi.subs(k=0))
