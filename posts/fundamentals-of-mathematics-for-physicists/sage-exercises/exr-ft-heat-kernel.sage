# Exercise exr-ft-heat-kernel: the heat kernel as the inverse transform of e^{-kappa k^2 t}, its properties, and the
# solution of the heat equation with a Gaussian initial profile by the convolution theorem.
x, k, t, kap = var("x k t kappa")
assume(t > 0); assume(kap > 0); assume(x, "real")
K = (integrate(exp(-kap * k ** 2 * t) * exp(I * k * x), k, -oo, oo) / (2 * pi)).simplify_full()
print("heat kernel K_t(x) =", K)
print("K solves u_t = kappa u_xx:", bool((diff(K, t) - kap * diff(K, x, 2)).simplify_full() == 0), "; unit integral:", bool(integrate(K, x, -oo, oo).canonicalize_radical() == 1))
u0hat = integrate(exp(-x ** 2 / 2) * exp(-I * k * x), x, -oo, oo).simplify_full()
uhat = exp(-kap * k ** 2 * t) * u0hat
u = (integrate(uhat * exp(I * k * x), k, -oo, oo) / (2 * pi)).simplify_full()
print("solution with u_0 = e^{-x^2/2}: u(t, x) =", u)
print("u solves the heat equation:", bool((diff(u, t) - kap * diff(u, x, 2)).simplify_full() == 0), "; u(0, x) = e^{-x^2/2}:", bool((u.subs(t=0) - exp(-x ** 2 / 2)).simplify_full() == 0))
print("u = (1 + 2 kappa t)^(-1/2) exp(-x^2/(2(1 + 2 kappa t))):", bool((u - exp(-x ** 2 / (2 * (1 + 2 * kap * t))) / sqrt(1 + 2 * kap * t)).canonicalize_radical() == 0))
