# Exercise exr-qm-stationary-phase-sage: int e^(-a^2) e^(i (a^2/2 + a^4/4)/hbar) da against sqrt(2 pi i hbar).
import math, cmath
for hb in [0.1, 0.03, 0.01]:
    f = lambda a: a ** 2 / 2 + a ** 4 / 4
    re = numerical_integral(lambda a: math.exp(-a * a) * math.cos(f(a) / hb), -7, 7, max_points=5000)[0]
    im = numerical_integral(lambda a: math.exp(-a * a) * math.sin(f(a) / hb), -7, 7, max_points=5000)[0]
    I_num = complex(re, im); I_sp = cmath.sqrt(2j * math.pi * hb)
    print("hbar =", hb, " numerical", I_num, " stationary phase", I_sp, " relative error", abs(I_num - I_sp) / abs(I_sp))
