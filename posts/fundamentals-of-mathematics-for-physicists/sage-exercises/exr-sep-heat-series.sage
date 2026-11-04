# Exercise exr-sep-heat-series: u_t = u_xx on [0, pi] with zero ends and u(0, x) = x(pi - x): the sine series solution
# sum_{n odd} 8/(pi n^3) e^(-n^2 t) sin(nx), checked on the partial sum N = 5.
t, x = var("t x")
phi = x * (pi - x)
b = lambda k: 2 / pi * integrate(phi * sin(k * x), x, 0, pi)
N = 5
u = sum(b(k) * exp(-k ** 2 * t) * sin(k * x) for k in range(1, N + 1))
print("partial sum u_5(t, x) =", u)
print("u_t = u_xx exactly:", bool((diff(u, t) - diff(u, x, 2)).simplify_full() == 0))
print("u(t, 0) = u(t, pi) = 0:", bool(u.subs(x=0) == 0), bool(u.subs(x=pi).simplify_full() == 0))
err = integrate((u.subs(t=0) - phi) ** 2, x, 0, pi)
print("squared error of the initial condition:", numerical_approx(err), "(relative", numerical_approx(err / integrate(phi ** 2, x, 0, pi)), ")")
print("value at the midpoint after t = 1:", numerical_approx(u.subs(t=1, x=pi / 2)), "against the first term alone:", numerical_approx(b(1) * exp(-1)))
