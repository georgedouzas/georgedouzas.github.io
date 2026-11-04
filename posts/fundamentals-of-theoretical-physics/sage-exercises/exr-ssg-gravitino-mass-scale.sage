# Exercise exr-ssg-gravitino-mass-scale: the gravitino mass m_{3/2} = <F>/(sqrt(3) M_P) with the reduced Planck mass
# M_P = 2.435e18 GeV (Martin, hep-ph/9709356, eq. 7.5.4, with <F> the F-term vacuum value of mass dimension 2), for
# sqrt(<F>) = 1e5 GeV (gauge mediation), 1e11 GeV (gravity mediation) and for the O'Raifeartaigh vacuum energy V = k^2.
MP = 2.435e18
for sF in (1e5, 1e11):
    m32 = sF^2 / (sqrt(3.0) * MP)
    print("sqrt(F) = %.0e GeV: m_3/2 = %.3e GeV" % (sF, m32))
print("[PASS] sqrt(F) = 100 TeV gives m_3/2 of a few eV (2.37 eV), gauge mediation:", abs(1e10 / (sqrt(3.0) * MP) * 1e9 - 2.371) < 0.01)
print("[PASS] sqrt(F) = 1e11 GeV gives m_3/2 of a few TeV (2.37 TeV), gravity mediation:", abs(1e22 / (sqrt(3.0) * MP) / 1e3 - 2.371) < 0.01)
k = var("k", domain="positive"); MPs = var("M_P", domain="positive")
print("[PASS] in the O'Raifeartaigh model F = k, V = k^2 = 3 m_3/2^2 M_P^2 (the vacuum energy that supergravity can cancel with -3|W|^2/M_P^2):",
      bool((k^2 - 3 * (k / (sqrt(3) * MPs))^2 * MPs^2).simplify_full() == 0))
