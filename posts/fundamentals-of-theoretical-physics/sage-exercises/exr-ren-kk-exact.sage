# Exercise exr-ren-kk-exact: the proper-time integral of the theta function, int_{r/L^2}^{r} dt/t sum_n e^(-t n^2), R = 1,
# r = pi/X_1^2 = pi/4, against the approximation 2 X_1 (L - 1).
import mpmath as mpm
r = mpm.pi / 4
th = lambda t: mpm.jtheta(3, 0, mpm.e**(-t))
for L in [10, 100]:
    ex = mpm.quad(lambda t: th(t) / t, [r / L**2, 0.01, r])
    print("Lambda = %3d: exact %.9f, approximation %d, difference %.3e" % (L, ex, 4 * (L - 1), ex - 4 * (L - 1)))
print("e^(-4 pi) = %.3e" % mpm.e**(-4 * mpm.pi))
