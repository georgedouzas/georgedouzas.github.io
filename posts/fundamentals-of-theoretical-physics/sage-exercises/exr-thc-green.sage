# Exercise exr-thc-green: the periodic Green's function of -d_tau^2 + omega^2 on the circle of circumference beta,
# from its Fourier (Matsubara) series G(tau) = (1/beta) sum_n e^{-i omega_n tau}/(omega_n^2 + omega^2), against the closed
# form cosh(omega(tau - beta/2))/(2 omega sinh(beta omega/2)) on 0 <= tau <= beta; the Fourier coefficients of the closed form.
tau, beta, w, n = var("tau beta omega n"); assume(beta > 0, w > 0)
G = cosh(w * (tau - beta / 2)) / (2 * w * sinh(beta * w / 2))
Ge = (exp(w * (tau - beta / 2)) + exp(-w * (tau - beta / 2))) / (2 * w * (exp(beta * w / 2) - exp(-beta * w / 2)))   # same G, exponentials
for k in [0, 1, 2, -3]:
    wk = 2 * pi * k / beta
    c = integrate(Ge * exp(I * wk * tau), tau, 0, beta).simplify_full()
    print("n = %2d: int_0^beta G e^{i omega_n tau} d tau = 1/(omega_n^2 + omega^2):" % k, bool((c - 1 / (wk ** 2 + w ** 2)).simplify_full() == 0))
bv, wv = 1.5, 0.8
for tv in [0.0, 0.4, 1.1]:
    s = sum(cos(2 * pi * k * tv / bv) / ((2 * pi * k / bv) ** 2 + wv ** 2) for k in range(-20000, 20001)).n() / bv
    print("beta = %s, omega = %s, tau = %s: series %.7f, closed form %.7f" % (bv, wv, tv, s, G.subs({tau: tv, beta: bv, w: wv}).n()))
