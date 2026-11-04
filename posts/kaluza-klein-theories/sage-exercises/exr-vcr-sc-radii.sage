# Exercise exr-vcr-sc-radii: Freund-Rubin vacua AdS_p x S^q supported by a p-form flux F = c eps_p. In the mostly-plus
# form of the Einstein equations R_MN = (1/(2 (p-1)!)) (F_M... F_N^... - (p-1)/(p (D-2)) g_MN F^2) the flux gives
# R_mu nu = -(q-1) c^2/(2(D-2)) g and R_mn = (p-1) c^2/(2(D-2)) g; with R_mu nu = -(p-1)/L^2 g and R_mn = (q-1)/a^2 g
# the radii follow. Tabulate a, L and a/L.
c = var("c", domain="positive")
for p, q in [(2, 2), (3, 3), (4, 7), (5, 5), (7, 4), (2, 3), (3, 7)]:
    D = p + q
    L2 = (p - 1) / (c ^ 2 * (q - 1) / (2 * (D - 2))); a2 = (q - 1) / (c ^ 2 * (p - 1) / (2 * (D - 2)))
    print("AdS_%d x S^%d: L = %s, a = %s, a/L = %s" % (p, q, sqrt(L2), sqrt(a2), sqrt(a2 / L2)))
