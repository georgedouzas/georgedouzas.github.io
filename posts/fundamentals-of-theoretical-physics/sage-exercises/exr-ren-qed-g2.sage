# Exercise exr-ren-qed-g2: the coefficient C4 of (alpha/pi)^2 in g/2 - 1, estimated from the measured value.
al = 1 / 137.035999; a_meas = 0.00115965218059
r = al / pi
C4 = (a_meas - r / 2) / r^2
print("alpha/2pi = %.10f, measured %.10f, difference %.4e" % ((r / 2).n(), a_meas, (a_meas - r / 2).n()))
print("C4 estimate = %.5f" % C4.n())
print("effect of a third-order term 1.2 (alpha/pi)^3 on the estimate: %.4f" % (1.2 * r).n())
C4_exact = 197 / 144 + pi^2 / 12 - pi^2 / 2 * log(2) + 3 / 4 * zeta(3)
print("closed-form C4 = %.9f" % C4_exact.n())
