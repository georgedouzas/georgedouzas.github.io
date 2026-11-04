# Exercise exr-kkf-kink-norm: normalisation of the orbifold zero mode f_0 = sqrt(2M/(1 - exp(-2 pi M R))) exp(-M y) on
# [0, pi R], the fraction of the norm within y < R/10, and the limit M -> 0.
y, M, R = var("y M R"); assume(M > 0, R > 0)
f0 = sqrt(2 * M / (1 - exp(-2 * pi * M * R))) * exp(-M * y)
print("int_0^{pi R} f_0^2 dy =", integrate(f0 ** 2, y, 0, pi * R).simplify_full())
frac = (integrate(f0 ** 2, y, 0, R / 10) / integrate(f0 ** 2, y, 0, pi * R)).simplify_full()
print("fraction within R/10 =", frac)
for MR in (1, 10, 100):
    print("  M R = %d: %.6f" % (MR, frac.subs(M=MR / R).simplify_full().n()))
print("limit M -> 0 of f_0^2 =", limit(f0 ** 2, M=0), "= 1/(pi R)")
