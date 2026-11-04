# Exercise exr-lgt-wa-temperature: the temperature of a lattice with N_t sites in periodic Euclidean time.
hbarc = 0.1973269804                         # GeV fm
for a in (0.1, 0.05):
    for Nt in (4, 6, 8):
        print('a = %.2f fm, N_t = %d: T = hbar c / (N_t a) = %.0f MeV' % (a, Nt, 1000 * hbarc / (Nt * a)))
print('N_t needed at a = 0.05 fm for T = 160 MeV:', (hbarc / (0.160 * 0.05)).n(digits=4))
