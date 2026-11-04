# Exercise exr-tsb-xy-chain (Boudreau-Swanson, Ch. 20, Exercise 11): the classical XY chain of L planar spins with
# H = -J sum_{i=1}^{L-1} cos(theta_i - theta_{i+1}) and free ends. Z = (2 pi)^L I_0(beta J)^(L-1), the mean energy
# -(L-1) J I_1/I_0, and the correlation <cos(theta_1 - theta_{1+r})> = (I_1/I_0)^r.
K = 1.3   # beta J
f = lambda t: exp(K * cos(t))
I0 = bessel_I(0, K).n(); I1 = bessel_I(1, K).n()
Z2 = (2 * pi * numerical_integral(lambda t: f(t), 0, 2 * pi)[0]).n()
print("L = 2: Z by integration %.8f, (2 pi)^2 I0 = %.8f" % (Z2, (2 * pi)^2 * I0))
# L = 3 by a two-dimensional numerical integral over the relative angles (the overall rotation gives 2 pi)
from scipy import integrate as si
import math
Z3 = 2 * math.pi * si.dblquad(lambda u, v: math.exp(K * math.cos(u)) * math.exp(K * math.cos(v)), 0, 2 * math.pi, 0, 2 * math.pi)[0]
print("L = 3: Z by integration %.6f, (2 pi)^3 I0^2 = %.6f" % (Z3, (2 * pi)^3 * I0^2))
c1 = numerical_integral(lambda t: cos(t) * f(t), 0, 2 * pi)[0] / numerical_integral(f, 0, 2 * pi)[0]
print("<cos> per bond %.10f, I1/I0 = %.10f" % (c1, I1 / I0))
for r in (1, 2, 5):
    print("correlation at distance %d: (I1/I0)^%d = %.6f" % (r, r, (I1 / I0)^r))
print("correlation length 1/log(I0/I1) at beta J = 1.3: %.4f" % (1 / log(I0 / I1)))
