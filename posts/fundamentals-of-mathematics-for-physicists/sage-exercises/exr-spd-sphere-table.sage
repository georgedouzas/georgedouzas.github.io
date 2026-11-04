# Exercise exr-spd-sphere-table: spinor data for the unit spheres S^2, ..., S^8.
print(" n  gamma size  Majorana (n,0)  lowest |lambda|  multiplicity  A-hat  zero modes from a gauge field")
for n in range(2, 9):
    N = 2 ^ (n // 2)
    maj = "yes" if n % 8 in (0, 1, 2, 6, 7) else "no"
    gauge = "yes (int ch_%d != 0 possible)" % (n // 2) if n % 2 == 0 else "no chirality (odd n)"
    print("%2d %11d %15s %16s %13d %6d   %s" % (n, N, maj, QQ(n) / 2, N, 0, gauge))
