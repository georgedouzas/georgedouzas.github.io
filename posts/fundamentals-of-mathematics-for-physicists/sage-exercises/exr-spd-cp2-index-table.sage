# Exercise exr-spd-cp2-index-table: the index (Q + 1)(Q + 2)/2 of the spin-c Dirac operator of CP^2 and the number of
# zero modes (|Y| + 2)(|Y| + 4)/8 from SU(3) Casimirs (Dolan 2003, eqs. (55)-(56)).
print("  Q  index  zero modes")
for Q in range(-7, 6):
    ind = (Q + 1) * (Q + 2) / 2
    if Q in (-1, -2):
        zm = 0
    else:
        Y = 2 * Q if Q >= 0 else 2 * Q + 6
        zm = (abs(Y) + 2) * (abs(Y) + 4) / 8
    print("%3d  %5s  %10s  %s" % (Q, ind, zm, "agree" if zm == abs(ind) else "DIFFER"))
