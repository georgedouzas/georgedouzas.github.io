# Exercise exr-log-power-branches: the principal square root jumps by a sign across the negative real axis; the two
# values of z^(1/2) = exp((1/2) log z) for the two nearest branches of log.
for z in [CC(-1, 0.01), CC(-1, -0.01)]:
    print("z =", z, "  principal sqrt(z) =", z.sqrt(), "  Arg z/2 =", z.arg()/2)
z0 = CC(-1, 0.01)
branches = [exp((z0.log() + 2*pi*I*k)/2) for k in (0, 1)]
print("the two values of z^(1/2) from log z + 2 pi i k, k = 0, 1:", branches)
print("they are negatives of each other:", abs(branches[0] + branches[1]) < 1e-12)
