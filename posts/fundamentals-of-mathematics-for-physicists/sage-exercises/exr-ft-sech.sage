# Exercise exr-ft-sech: numerical check of the transform of sech x, pi sech(pi k/2), at several k.
x = var("x")
for kv in [0, 1/2, 1, 2, 3]:
    num = numerical_integral(lambda t: cos(kv * t) / cosh(t), -40, 40)[0]
    exact = (pi / cosh(pi * kv / 2)).n()
    print("k =", kv, ": int sech(x) cos(kx) dx =", "%.10f" % num, "; pi sech(pi k/2) =", "%.10f" % exact, "; agree to 1e-8:", bool(abs(num - exact) < 1e-8))
