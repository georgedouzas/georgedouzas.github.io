# Exercise exr-ren-lattice-c00: the lattice tadpole constants C_d = int_0^oo (e^(-2t) I_0(2t))^d dt for d = 3, 4.
import mpmath as mpm
mpm.mp.dps = 20
for d in [3, 4]:
    C = mpm.quad(lambda u: (mpm.exp(-2 * u) * mpm.besseli(0, 2 * u))**d, [0, 1, 10, 100, 1000, mpm.inf])
    print("C_%d = %s" % (d, mpm.nstr(C, 10)))
print("large-t behaviour: e^(-2t) I_0(2t) sqrt(4 pi t) at t = 1000:", mpm.nstr(mpm.exp(-2000) * mpm.besseli(0, 2000) * mpm.sqrt(4 * mpm.pi * 1000), 8))
