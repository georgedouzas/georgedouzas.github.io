# Exercise exr-qm-born-box-energy: energy probabilities of psi = sqrt(30/L^5) x (L - x) in the box, and <H> two ways.
x = var("x", domain="real"); hbar, m, L = var("hbar m L", domain="positive")
psi = sqrt(30 / L ** 5) * x * (L - x)
phi = lambda n: sqrt(2 / L) * sin(n * pi * x / L)
E = lambda n: n ** 2 * pi ** 2 * hbar ** 2 / (2 * m * L ** 2)
probs = [(integrate(phi(n) * psi, x, 0, L) ** 2).canonicalize_radical() for n in range(1, 8)]
print("Prob(E_n), n = 1..7:", probs)
print("equal to 960/(n pi)^6 for odd n:", all(bool(probs[n - 1] == (960 / (n * pi) ** 6 if n % 2 else 0)) for n in range(1, 8)))
k = var("k")
print("sum over odd n of 960/(n pi)^6 =", SR(sum(960 / ((2 * k + 1) * pi) ** 6, k, 0, oo)).simplify_full(), "; Prob(E_1) =", (960 / pi ** 6).n(digits=6))
H_direct = integrate(psi * (-hbar ** 2 / (2 * m)) * diff(psi, x, 2), x, 0, L).simplify_full()
H_series = SR(sum(960 / ((2 * k + 1) * pi) ** 6 * (2 * k + 1) ** 2 * pi ** 2 * hbar ** 2 / (2 * m * L ** 2), k, 0, oo)).simplify_full()
print("<H> directly =", H_direct, "; from the probabilities =", H_series, "; <H>/E_1 =", (H_direct / E(1)).simplify_full(), "=", (H_direct / E(1)).n(digits=5))
