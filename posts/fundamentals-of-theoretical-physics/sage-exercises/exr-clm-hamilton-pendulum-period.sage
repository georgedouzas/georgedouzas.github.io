# Exercise exr-clm-hamilton-pendulum-period: the period of the pendulum H = p^2/2 - cos q as a function of the
# amplitude q0 (integral by tanh-sinh quadrature, which handles the endpoint singularity): T = 4 int_0^q0 dq/sqrt(2(cos q - cos q0)) = 4 K(sin^2(q0/2)), compared with 2 pi.
import mpmath
mpmath.mp.dps = 20
for q0 in [0.1, 1.0, pi / 2, 2.5, 3.0, 3.13]:
    c0 = mpmath.cos(float(q0)); T_int = 4 * float(mpmath.quad(lambda s: 1 / mpmath.sqrt(abs(2 * (mpmath.cos(s) - c0))), [0, float(q0)]))
    T_K = 4 * elliptic_kc(sin(q0 / 2)**2)
    print("q0 = %.3f: T (integral) = %.6f, 4K = %.6f, T/(2 pi) = %.4f" % (float(q0), T_int, float(T_K), float(T_K / (2 * pi))))
