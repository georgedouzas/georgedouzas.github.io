# Exercise exr-vcp-add-emission: the summed graviton emission rate (E R)^n/M_Pl^2 = E^n/M^{n+2} (Arkani-Hamed,
# Dimopoulos, Dvali, eqs. 5-6) and the branching ratio (E/M)^{n+2} (eq. 7) for processes at energy E: tiny at
# E = 20 MeV (supernova core) and order one at E ~ M (collisions at the fundamental scale).
E, R, M, n = var("E R M n", domain="positive")
print("(E R)^n/(M^{n+2} R^n) - E^n/M^{n+2} =", ((E * R)**n / (M**(n + 2) * R**n) - E**n / M**(n + 2)).canonicalize_radical())
for nn in [2, 3, 6]:
    for EE, lab in [(0.02, "supernova, 20 MeV"), (100.0, "100 GeV"), (1000.0, "1 TeV")]:
        print("n = %d, M = 1 TeV, E = %s: branching ratio (E/M)^(n+2) = %.1e" % (nn, lab, (EE / 1000.0)**(nn + 2)))
