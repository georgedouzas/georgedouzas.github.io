# Exercise exr-de-string-greens: the string on [0, pi]: (a) the tent Green's function G = x_<(pi - x_>)/pi and its sine
# coefficients 2 sin(n xi)/(pi n^2); (b) the static shape under f = 1; (c) the standing-wave series released from it;
# (d) the energy of the partial sums is constant.
x, xi, t = var("x xi t")
assume(xi > 0, xi < pi, x > 0, x < pi)
Ga, Gb = x * (pi - xi) / pi, xi * (pi - x) / pi
print("(a) slope jump at x = xi:", (diff(Gb, x) - diff(Ga, x)).subs(x=xi).simplify_full(), "| continuity:", bool((Ga - Gb).subs(x=xi) == 0))
coef = lambda n: (2 / pi * (integrate(Ga * sin(n * x), x, 0, xi) + integrate(Gb * sin(n * x), x, xi, pi))).simplify_full()   # x < xi on [0, xi], x > xi on [xi, pi]
print("    sine coefficients of G(., xi) for n = 1, 2, 3:", [coef(n) for n in range(1, 4)], "| equal 2 sin(n xi)/(pi n^2):", all(bool((coef(n) - 2 * sin(n * xi) / (pi * n ** 2)).simplify_full() == 0) for n in range(1, 4)))
u0 = (integrate(Gb, xi, 0, x) + integrate(Ga, xi, x, pi)).simplify_full()   # int_0^pi G(x, xi) * 1 dxi: xi < x uses Gb, xi > x uses Ga
print("(b) static shape under f = 1:", u0, "| -u'' = 1 and u(0) = u(pi) = 0:", bool(-diff(u0, x, 2) == 1), bool(u0.subs(x=0) == 0), bool(u0.subs(x=pi) == 0))
b = lambda n: (2 / pi * integrate(u0 * sin(n * x), x, 0, pi)).simplify_full()
N = 5
u = sum(b(n) * sin(n * x) * cos(n * t) for n in range(1, N + 1))
print("(c) b_1..b_5:", [b(n) for n in range(1, 6)], "| u_tt = u_xx:", bool((diff(u, t, 2) - diff(u, x, 2)).simplify_full() == 0), "| u_t(0, x) = 0:", bool(diff(u, t).subs(t=0) == 0))
energy = (integrate(diff(u, t) ** 2 + diff(u, x) ** 2, x, 0, pi) / 2).simplify_full()
print("(d) energy of the partial sum:", energy, "| constant in t:", bool(diff(energy, t) == 0), "| equals (pi/4) sum n^2 b_n^2:", bool((energy - pi / 4 * sum(n ** 2 * b(n) ** 2 for n in range(1, N + 1))).simplify_full() == 0))
