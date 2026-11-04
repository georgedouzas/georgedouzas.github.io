# Exercise exr-fpi-oscillator-partition: Z = sum_n e^{-beta w (n + 1/2)} = 1/(2 sinh(beta w/2)) and
# E = -d log Z/d beta = (w/2) coth(beta w/2) = w (n_B + 1/2).
n = var("n"); b, w = var("b w", domain="positive")
Z = sum(exp(-b * w * (n + QQ(1) / 2)), n, 0, oo)
print("Z =", Z.simplify_full(), "  equals 1/(2 sinh(b w/2)):", bool((Z - 1 / (2 * sinh(b * w / 2))).exponentialize().simplify_full() == 0))
E = -diff(log(Z), b)
nB = 1 / (exp(b * w) - 1)
print("E = w (n_B + 1/2):", bool((E - w * (nB + QQ(1) / 2)).simplify_full() == 0), "  E = (w/2) coth(b w/2):", bool((E - w / 2 * coth(b * w / 2)).exponentialize().simplify_full() == 0))
