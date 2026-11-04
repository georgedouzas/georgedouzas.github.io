# Exercise exr-hta-sphere-numeric: the heat trace of the unit sphere against 1/t + 1/3 + t/15, the remainder divided by t^2
# compared with 4/315.
RF = RealField(100)
for t in [QQ(1) / 5, QQ(1) / 10, QQ(1) / 20, QQ(1) / 40]:
    Z = sum((2 * l + 1) * exp(-RF(t) * l * (l + 1)) for l in range(301))
    r = Z - (1 / RF(t) + RF(1) / 3 + RF(t) / 15)
    print("t = %s: Z = %.10f  remainder/t^2 = %.6f" % (t, Z, r / RF(t) ** 2))
print("4/315 =", (4 / 315).n(digits=6))
