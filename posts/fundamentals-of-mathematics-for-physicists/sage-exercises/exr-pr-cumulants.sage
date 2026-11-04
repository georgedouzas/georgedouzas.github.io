# Exercise exr-pr-cumulants: cumulants kappa_n = K^(n)(0), K(s) = log E e^{sX}, for the Poisson distribution (all equal
# to lambda), for the die (kappa_3 = 0, kappa_4 = -259/24) and for the Gaussian (zero beyond the second).
s, lam, mu, sigma = var("s lam mu sigma")
assume(lam > 0); assume(sigma > 0)
K_pois = lam * (exp(s) - 1)
print("Poisson cumulants kappa_1..kappa_5:", [diff(K_pois, s, n).subs(s=0) for n in range(1, 6)])
M_die = sum(exp(s * k) for k in range(1, 7)) / 6
K_die = log(M_die)
kap = [diff(K_die, s, n).subs(s=0).simplify_full() for n in range(1, 5)]
print("die cumulants kappa_1..kappa_4:", kap)
print("kappa_2 = 35/12:", bool(kap[1] == QQ(35) / 12), "; kappa_4 = E(X - 7/2)^4 - 3 sigma^4 = 707/48 - 1225/48:", bool(kap[3] == QQ(707) / 48 - QQ(1225) / 48))
K_gauss = mu * s + sigma ** 2 * s ** 2 / 2
print("Gaussian cumulants kappa_1..kappa_5:", [diff(K_gauss, s, n).subs(s=0) for n in range(1, 6)])
x = var("x")
assume(x, "real")
M_gauss = integrate(exp(s * x) * exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt(2 * pi) * sigma), x, -oo, oo)
print("Gaussian E e^{sX} = ", M_gauss.simplify_full(), "; log is quadratic:", bool((log(M_gauss) - K_gauss).simplify_full() == 0))
