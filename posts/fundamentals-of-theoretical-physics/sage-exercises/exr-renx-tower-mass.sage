# Exercise exr-renx-tower-mass: the radius-dependent part of the coincident propagator on R^4 x S^1 (l = 1), from the mode sum
# with a proper-time cutoff, (1/l) sum_n e^(-t (2 pi n)^2) minus its continuum part, against the image sum zeta(3)/(4 pi^2).
import mpmath as mpm
th = lambda t: mpm.jtheta(3, 0, mpm.e**(-4 * mpm.pi**2 * t))
def F(Lam):
    f = lambda t: (th(t) - 1 / mpm.sqrt(4 * mpm.pi * t)) / (16 * mpm.pi**2 * t**2)
    return mpm.quad(f, [1 / Lam**2, 0.001, 0.01, 0.1, 1, mpm.inf])
for Lam in [5, 10, 20]:
    print("cutoff %2d: %.10f" % (Lam, F(Lam)))
print("image sum 2 zeta(3)/(8 pi^2) = %.10f" % (2 * zeta(3) / (8 * pi^2)).n())
print("delta m^2 = lambda_4 zeta(3)/(8 pi^2 l^2): coefficient %.6f" % (zeta(3) / (8 * pi^2)).n())
