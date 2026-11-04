# Exercise exr-ds-weyl-counting: N(lambda), the number of lattice points (m, n) with m^2 + n^2 <= lambda, against Weyl's
# prediction pi lambda for the square torus, for lambda = 10^2, 10^3, 10^4.
def N_count(lam):
    r = isqrt(lam)
    return sum(1 for m in range(-r, r + 1) for n in range(-r, r + 1) if m ** 2 + n ** 2 <= lam)
for lam in [10 ** 2, 10 ** 3, 10 ** 4]:
    Nl = N_count(lam); W = (pi * lam).n()
    print("lambda = %6d: N(lambda) = %6d   pi lambda = %10.1f   relative error = %.5f" % (lam, Nl, W, abs(Nl - W) / W))
