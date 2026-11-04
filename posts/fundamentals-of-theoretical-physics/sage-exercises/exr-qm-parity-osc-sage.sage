# Exercise exr-qm-parity-osc-sage: parity of the Hermite functions, [H, Pi] = 0, and <Pi> in a superposition.
xi = var("xi", domain="real")
psi = lambda n, X=xi: pi ** (-1 / 4) / sqrt(2 ** n * factorial(n)) * hermite(n, X) * exp(-X ** 2 / 2)
print("psi_n(-xi) = (-1)^n psi_n(xi), n = 0..8:", all(bool((psi(n, -xi) - (-1) ** n * psi(n)).simplify_full() == 0) for n in range(9)))
f = function("f")
H = lambda g: (-diff(g, xi, 2) + xi ** 2 * g) / 2
Pf = f(-xi)
print("H Pi f - Pi H f =", (H(Pf) - H(f(xi)).subs(xi=-xi)).simplify_full())
st = (psi(0) + psi(1) + psi(2)) / sqrt(3)
print("<Pi> in (psi_0 + psi_1 + psi_2)/sqrt 3 =", integrate(st * st.subs(xi=-xi), xi, -oo, oo).simplify_full())
