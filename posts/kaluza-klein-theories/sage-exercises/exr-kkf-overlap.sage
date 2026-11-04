# Exercise exr-kkf-overlap: overlap of a zero mode localised at y = 0 (profile exp(-M y)) with one localised at y = pi R
# (profile exp(M (y - pi R))), both normalised on [0, pi R]; it falls like pi M R exp(-pi M R) for large M R.
y, M, R = var("y M R"); assume(M > 0, R > 0)
Nf = sqrt(2 * M / (1 - exp(-2 * pi * M * R)))
ov = integrate(Nf * exp(-M * y) * Nf * exp(M * (y - pi * R)), y, 0, pi * R).simplify_full()
print("overlap =", ov)
print("closed form 2 M pi R exp(-pi M R)/(1 - exp(-2 pi M R)):", bool(ov == 2 * M * pi * R * exp(-pi * M * R) / (1 - exp(-2 * pi * M * R))))
for MR in (1, 5, 10):
    v = ov.subs(M=MR / R).n()
    print("  M R = %2d: overlap %.3e, ratio to 2 pi M R exp(-pi M R) = %.6f" % (MR, v, v / (2 * pi * MR * exp(-pi * MR)).n()))
