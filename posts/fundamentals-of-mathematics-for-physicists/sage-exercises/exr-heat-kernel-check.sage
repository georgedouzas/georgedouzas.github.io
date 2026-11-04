# Exercise exr-heat-kernel-check: the heat kernel solves u_t = kappa u_xx, has unit integral and variance 2 kappa t, and
# its convolution with e^(-y^2) is a Gaussian of variance 1/2 + 2 kappa t.
t, x, y, kap = var("t x y kap")
assume(t > 0, kap > 0)
K = exp(-x ** 2 / (4 * kap * t)) / sqrt(4 * pi * kap * t)
print("K_t - kappa K_xx = 0:", bool((diff(K, t) - kap * diff(K, x, 2)).simplify_full() == 0))
print("integral of K:", integrate(K, x, -oo, oo).canonicalize_radical())
print("integral of x^2 K:", integrate(x ** 2 * K, x, -oo, oo).canonicalize_radical())
u = integrate(K.subs(x=x - y) * exp(-y ** 2), y, -oo, oo).canonicalize_radical()
print("convolution with e^(-y^2):", u)
var_u = (integrate(x ** 2 * u, x, -oo, oo) / integrate(u, x, -oo, oo)).canonicalize_radical()
print("its variance:", var_u, "= 1/2 + 2 kappa t:", bool((var_u - (1 / 2 + 2 * kap * t)).simplify_full() == 0))
print("the convolution solves the heat equation:", bool((diff(u, t) - kap * diff(u, x, 2)).simplify_full() == 0))
