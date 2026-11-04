# Exercise exr-argument-values: moduli and principal arguments of -1 - i, 1 - i sqrt(3), -2 i and (1 + i)^10.
for z in [-1 - I, 1 - I*sqrt(3), -2*I, (1 + I)^10]:
    z = SR(z)
    print("z =", z.expand(), "  |z| =", abs(z).simplify_full(), "  Arg z =", arg(z).simplify_full())
print("(1 + i)^10 = 32 i:", bool(SR((1 + I)^10).expand() == 32*I))
