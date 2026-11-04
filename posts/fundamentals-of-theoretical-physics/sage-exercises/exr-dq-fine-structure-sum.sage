# Exercise exr-dq-fine-structure-sum: first-order kinetic, spin-orbit and Darwin shifts in hydrogen-like atoms (units m (Z alpha)^4)
# summed for all (n, l, j) with n <= 4 and compared with -(1/(2 n^4)) (n/(j + 1/2) - 3/4).
def shifts(n, l, j):
    En = -QQ(1) / (2 * n^2)                      # Bohr energy in units m (Z alpha)^2; below all in units m (Z alpha)^4
    r1, r2 = QQ(1) / n^2, QQ(1) / (n^3 * (l + QQ(1) / 2))
    kin = -QQ(1) / 2 * (En^2 + 2 * En * r1 + r2)     # -<p^4>/8m^3 = -<(E - V)^2>/2m with V = -1/r
    if l == 0:
        so, dar = 0, QQ(1) / (2 * n^3)
    else:
        r3 = QQ(1) / (n^3 * l * (l + QQ(1) / 2) * (l + 1))
        SL = (j * (j + 1) - l * (l + 1) - QQ(3) / 4) / 2
        so, dar = QQ(1) / 2 * r3 * SL, 0            # (Z alpha / 2 m^2) <r^-3> S.L
    return kin, so, dar
ok = True
for n in range(1, 5):
    for l in range(n):
        for j in ([QQ(1) / 2] if l == 0 else [l - QQ(1) / 2, l + QQ(1) / 2]):
            kin, so, dar = shifts(n, l, j)
            tot = kin + so + dar; dirac = -QQ(1) / (2 * n^4) * (n / (j + QQ(1) / 2) - QQ(3) / 4)
            ok = ok and tot == dirac
            print("n=%d l=%d j=%s: kinetic %s, spin-orbit %s, Darwin %s, total %s, formula %s" % (n, l, j, kin, so, dar, tot, dirac))
print("all totals agree with the formula:", ok)
