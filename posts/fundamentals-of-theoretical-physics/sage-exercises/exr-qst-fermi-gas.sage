# Exercise exr-qst-fermi-gas: the ideal Fermi gas at T = 0 (spin 1/2, hbar = 1, mass m): N, E, the degeneracy pressure
# p = -dE/dV at fixed N, and numbers for the conduction electrons of copper (n = 8.47e28 m^-3).
k, kF, V, m, N = var("k k_F V m N"); assume(k > 0, kF > 0, V > 0, m > 0, N > 0)
Nk = 2 * V / (2 * pi) ** 3 * integrate(4 * pi * k ** 2, k, 0, kF)
Ek = 2 * V / (2 * pi) ** 3 * integrate(4 * pi * k ** 2 * k ** 2 / (2 * m), k, 0, kF)
eF = kF ** 2 / (2 * m)
print("N =", Nk, "; E =", Ek, "; E/(N eps_F) =", (Ek / (Nk * eF)).simplify_full())
kF_N = (3 * pi ** 2 * N / V) ** (1 / 3)
E_NV = (3 / 5 * N * eF).subs({kF: kF_N})
p = (-diff(E_NV, V)).simplify_full()
print("p = -dE/dV =", p, "; p V/E =", (p * V / E_NV).simplify_full())
hbar, me, eV = 1.054571817e-34, 9.1093837e-31, 1.602176634e-19
nCu = 8.47e28
kFn = (3 * pi ** 2 * nCu) ** (1 / 3)
EF = (hbar * kFn) ** 2 / (2 * me)
print("copper: k_F = %.3e m^-1, eps_F = %.2f eV, T_F = %.3e K, p = %.2e Pa" % (kFn.n(), (EF / eV).n(), (EF / 1.380649e-23).n(), (2 / 5 * nCu * EF).n()))
