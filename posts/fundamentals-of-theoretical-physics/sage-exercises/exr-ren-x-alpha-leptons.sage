# Exercise exr-ren-x-alpha-leptons: 1/alpha_eff(M_Z) with the e, mu, tau loops (exact one-loop) and with leading logs.
al = 1 / 137.035999; MZ = 91.1876; leptons = {"e": 0.000510999, "mu": 0.1056584, "tau": 1.77686}
x = var('x'); exact, ll = 1 / al, 1 / al
for name, m in leptons.items():
    I = numerical_integral(x * (1 - x) * log(1 + x * (1 - x) * MZ^2 / m^2), 0, 1)[0]
    exact -= 2 / pi * I; ll -= 1 / (3 * pi) * (log(MZ^2 / m^2) - 5 / 3)
    print("after %-3s: exact %.4f   leading logarithms %.4f" % (name, exact.n(), ll.n()))
print("quarks must supply Delta(1/alpha) = %.2f to reach 128.9" % (exact - 128.9).n())
