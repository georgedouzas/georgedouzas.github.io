# Exercise exr-spd-hypersurface-ahat: Chern classes, Euler characteristic, p1 and A-hat of a smooth surface of degree d in
# CP^3, with c(TS) = (1 + x)^4 / (1 + d x) and int_S x^2 = d.
x = var("x")
print(" d  c1    chi   int p1   A-hat   spin")
for d in range(1, 9):
    c = ((1 + x) ^ 4 / (1 + d * x)).taylor(x, 0, 2)
    c1, c2 = c.coefficient(x, 1), c.coefficient(x, 2)
    chi = d * c2; p1 = d * (c1 ^ 2 - 2 * c2); ahat = -p1 / 24
    print("%2d %3s %6s %8s %7s   %s" % (d, c1, chi, p1, ahat, "yes" if ZZ(c1) % 2 == 0 else "no"))
print("closed form A-hat = d (d^2 - 4)/24:", all(-(d * (((4 - d) ^ 2) - 2 * (6 - 4 * d + d ^ 2)))/24 == d * (d ^ 2 - 4) / 24 for d in range(1, 9)))
