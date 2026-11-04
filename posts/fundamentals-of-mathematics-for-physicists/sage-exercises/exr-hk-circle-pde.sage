# Exercise exr-hk-circle-pde: each mode e^{-t n^2} e^{i n (theta - theta')} of the circle heat kernel solves the heat
# equation, the modes compose by the semigroup law, and the two Poisson forms of the heat trace agree to twenty digits.
t, s, th, thp, u = var("t s theta thetap u", domain="real")
ok = all(bool((diff(exp(-t * n ** 2) * exp(I * n * (th - thp)), t) - diff(exp(-t * n ** 2) * exp(I * n * (th - thp)), th, 2)).simplify_full() == 0) for n in range(-3, 4))
print("each mode solves d_t K = d_theta^2 K:", ok)
K = lambda tt, a, b, N: sum(exp(-tt * n ** 2) * exp(I * n * (a - b)) for n in range(-N, N + 1)) / (2 * pi)
lhs = integrate((K(t, th, u, 2) * K(s, u, thp, 2)).expand(), u, 0, 2 * pi)
print("semigroup on the modes |n| <= 2: int K_t K_s = K_{t+s}:", bool((lhs - K(t + s, th, thp, 2)).simplify_full() == 0))
RF = RealField(100)
for tt in [QQ(1) / 10, QQ(1) / 2, 1]:
    a = sum(exp(-RF(tt) * n ** 2) for n in range(-60, 61)); b = sqrt(RF(pi) / tt) * sum(exp(-RF(pi) ** 2 * k ** 2 / tt) for k in range(-10, 11))
    print("t =", tt, ":", a.n(digits=20), b.n(digits=20))
