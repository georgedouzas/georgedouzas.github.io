# Exercise exr-spd-flag-index: zero modes on SU(3)/(U(1) x U(1)) with monopole charges (T, Y), |T| >= |Y| (Dolan 2003,
# eq. (70)), against the index M (N^2 - M^2)/8 with M = T + 2, N = Y (eqs. (71)-(72)).
print("  T  Y  zero modes  index")
for T in range(0, 7):
    for Y in range(0, T + 1):
        if (T - Y) % 2: continue
        d = ((abs(T) + 2) ^ 2 - Y ^ 2) * (abs(T) + 2) / 8
        Mm, N = T + 2, Y
        ind = Mm * (N ^ 2 - Mm ^ 2) / 8
        print("%3d %2d %11s %6s %s" % (T, Y, d, ind, "agree up to sign" if abs(ind) == d else "DIFFER"))
