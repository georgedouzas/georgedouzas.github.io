# Exercise exr-ems-capacitance-noncoaxial (Macchi et al., Problem 2.15): two parallel cylinders, one inside the other.
# Line charges +lam at x = -A and -lam at x = +A; the equipotential s2/s1 = K is a circle with centre -A (K^2+1)/(K^2-1)
# and radius 2 A K/|K^2 - 1|. Fit two such circles to the cylinders and compare 2 pi/log(K1/K2) with the closed formula.
from scipy.optimize import fsolve
def cap_from_lines(R1, R2, c):
    f = lambda v: [2 * v[0] * v[1] / (v[1]**2 - 1) - R1, 2 * v[0] * v[2] / (v[2]**2 - 1) - R2,
                   v[0] * (v[2]**2 + 1) / (v[2]**2 - 1) - v[0] * (v[1]**2 + 1) / (v[1]**2 - 1) - c]
    A, K1, K2 = fsolve(f, [1.0, 3.0, 1.5])
    return 2 * pi.n() / log(K1 / K2)
for (R1, R2, c) in [(1.0, 2.0, 0.3), (1.0, 3.0, 1.5), (0.5, 2.0, 1.0)]:
    print("R1 = %s, R2 = %s, c = %s: from the lines %.6f, closed formula %.6f" % (R1, R2, c, cap_from_lines(R1, R2, c), 2 * pi.n() / arccosh((R1^2 + R2^2 - c^2) / (2 * R1 * R2)).n()))
print("c = 0, R1 = 1, R2 = 3: arccosh(10/6) - log(3) =", (arccosh(10 / 6) - log(3)).n())
