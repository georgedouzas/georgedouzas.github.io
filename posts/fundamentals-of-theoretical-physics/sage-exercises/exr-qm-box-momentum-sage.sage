# Exercise exr-qm-box-momentum-sage: <p>, <p^2> in the box eigenstates and the momentum wave function of phi_1.
x, p = var("x p", domain="real"); hbar, m, L = var("hbar m L", domain="positive")
phi = lambda n: sqrt(2 / L) * sin(n * pi * x / L)
for n in range(1, 6):
    mp = integrate(phi(n) * (-I * hbar) * diff(phi(n), x), x, 0, L).simplify_full()
    mp2 = integrate(phi(n) * (-hbar ** 2) * diff(phi(n), x, 2), x, 0, L).simplify_full()
    print("n =", n, " <p> =", mp, " <p^2>/(2m E_n) =", (mp2 / (n ** 2 * pi ** 2 * hbar ** 2 / L ** 2)).simplify_full())
pt = (integrate(exp(-I * p * x / hbar) * phi(1), x, 0, L) / sqrt(2 * pi * hbar)).simplify_full()
claimed = hbar * sqrt(pi * hbar * L) * (1 + exp(-I * p * L / hbar)) / (pi ** 2 * hbar ** 2 - p ** 2 * L ** 2)
print("phi~_1(p) matches hbar sqrt(pi hbar L)(1 + e^(-i p L/hbar))/(pi^2 hbar^2 - p^2 L^2):", bool((pt - claimed).canonicalize_radical() == 0))
dens = (4 * pi * hbar ** 3 * L * cos(p * L / (2 * hbar)) ** 2 / (pi ** 2 * hbar ** 2 - p ** 2 * L ** 2) ** 2).subs(hbar=1, L=1)
print("|phi~_1|^2 at p = 0:", dens.subs(p=0).n(digits=5), "; limit at p = pi:", limit(dens, p=pi).n(digits=5), "; at p = 2:", dens.subs(p=2).n(digits=5))
print("normalised:", numerical_integral(dens, -200, 200)[0])
