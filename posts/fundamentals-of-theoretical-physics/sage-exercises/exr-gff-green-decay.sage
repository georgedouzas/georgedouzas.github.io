# Exercise exr-gff-green-decay (Bauerschmidt-Brydges-Slade, eqs. 1.5.13-1.5.14 and Exercise 1.5.5): the massless lattice
# Green function of Z^3 on the axis, C_0x = int_0^oo e^{-6t} I_x(2t) I_0(2t)^2 dt, against 1/(4 pi |x|) (eta = 0); C_00 is
# Watson's integral and 1 - 1/(6 C_00) is the return probability of the simple random walk (Polya).
from scipy.special import ive
from scipy.integrate import quad
def C(x):
    f = lambda t: ive(x, 2 * t) * ive(0, 2 * t)^2
    return quad(f, 0, 50, limit=400)[0] + quad(f, 50, float('inf'), limit=400)[0]
C0 = C(0)
print("C_00 = %.6f, return probability 1 - 1/(6 C_00) = %.4f" % (C0, 1 - 1 / (6 * C0)))
for x in (1, 2, 4, 8, 16):
    c = C(x)
    print("x = %2d: C_0x = %.6f, 4 pi x C_0x = %.5f" % (x, c, 4 * pi.n() * x * c))
