# Exercise exr-emx-pol-gas: susceptibility of H2 from the atomic polarisability 4 pi (9/2) a0^3 per atom (SI).
a0 = 5.29177e-11
alpha_atom = 4 * pi * 9 / 2 * a0^3       # alpha/eps0 in m^3
alpha_mol = 2 * alpha_atom
N = 2.69e25
chi = N * alpha_mol
print('alpha/eps0 per molecule =', n(alpha_mol, digits=3), 'm^3')
print('chi_e = n alpha/eps0 =', n(chi, digits=3))
cm = chi / (1 - chi / 3)
print('Clausius-Mossotti chi =', n(cm, digits=6), '; relative correction =', n(cm / chi - 1, digits=3))
