# Exercise exr-lan-mean-field: solve the mean-field equation m = tanh(m T_c/T) at several temperatures and compare with
# the near-T_c form sqrt(3 (1 - T/T_c)) (T/T_c) and the low-temperature form 1 - 2 e^{-2 T_c/T}.
for t in [0.2, 0.3, 0.5, 0.7, 0.9, 0.99]:
    m = find_root(lambda x: x - tanh(x / t), 1e-6, 1.0)
    print("T/T_c = %.2f: m = %.6f, near-T_c form %.6f, low-T form %.6f" % (t, m, sqrt(3 * (1 - t)) * t, 1 - 2 * exp(-2 / t)))
x, t = var("x t")
print("series check: tanh(x/t) - x =", (tanh(x / t) - x).taylor(x, 0, 3))
