# Exercise exr-ren-x-tadpole-table: the quadratic coefficient and the coefficient of m^2 log m^2 of the tadpole (both
# times 16 pi^2) in five regularisations; q = m^2.
var('q L M', domain='positive')
import mpmath as mpm
forms = {"cutoff": (L^2 - q * log((L^2 + q) / q), L),
         "Pauli-Villars": (2 * log(2) * M^2 - q * log(4 * M^2 / q), M)}
Wt = -(L^4 / 2 - q * L^2 + q^2 / 2 * (log(L^2 / q) - euler_gamma + 3 / 2)) / 2      # 16 pi^2 W/V
forms["proper time"] = (2 * diff(Wt, q), L)
for name, (f, S) in forms.items():
    quad_coeff = f.expand().coefficient(S^2)
    log_coeff = limit(q * diff(f, q, 2), q=0)
    print("%-14s quadratic coefficient %-10s coefficient of m^2 log m^2: %s" % (name, quad_coeff, log_coeff))
print("%-14s quadratic coefficient %-10s coefficient of m^2 log m^2: %s" % ("dimensional", 0, 1))
C4 = mpm.quad(lambda u: (mpm.exp(-2 * u) * mpm.besseli(0, 2 * u))**4, [0, 1, 10, 100, mpm.inf])
print("%-14s quadratic coefficient %.4f (in units of 1/a^2)" % ("lattice", 16 * float(pi)**2 * float(C4)))
