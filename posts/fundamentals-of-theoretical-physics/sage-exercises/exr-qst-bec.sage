# Exercise exr-qst-bec: Bose-Einstein condensation of an ideal gas with the density of liquid helium-4:
# T_c = (2 pi hbar^2/(m k_B)) (n/zeta(3/2))^(2/3), and the condensate fraction 1 - (T/T_c)^(3/2).
print("zeta(3/2) =", zeta(3 / 2).n(digits=8), "; zeta(5/2) =", zeta(5 / 2).n(digits=8))
hbar, kB = 1.054571817e-34, 1.380649e-23
m = 4.002602 * 1.66053907e-27
n = 145.0 / m              # mass density 145 kg/m^3
Tc = 2 * pi * hbar ** 2 / (m * kB) * (n / zeta(3 / 2)) ** (2 / 3)
print("n = %.4e m^-3, T_c = %.3f K (the lambda point of helium-4 is 2.17 K)" % (n, Tc.n()))
t = var("t")
for tv in [0, 1 / 2, 0.9, 1]:
    print("T/T_c = %s: condensate fraction = %.4f" % (tv, (1 - t ** (3 / 2)).subs(t=tv).n()))
