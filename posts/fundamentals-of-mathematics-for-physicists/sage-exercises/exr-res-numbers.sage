# Exercise exr-res-numbers: resonance frequency, amplitudes and phase lags of the driven oscillator with omega0 = 1,
# gamma = 1/4, F = 1.
w = var("w")
w0, g, F = 1, QQ(1) / 4, 1
A = F / sqrt((w0 ** 2 - w ** 2) ** 2 + 4 * g ** 2 * w ** 2)
wr = sqrt(w0 ** 2 - 2 * g ** 2)
print("resonance frequency omega_r =", wr, "=", n(wr))
print("A(omega_r) =", A.subs(w=wr).canonicalize_radical(), "=", n(A.subs(w=wr)))
print("A(omega_0) =", A.subs(w=w0), "= 1/(2 gamma omega_0):", bool(A.subs(w=w0) == 1 / (2 * g * w0)))
print("the derivative of A^2 vanishes at omega_r:", bool(diff(A ** 2, w).subs(w=wr).simplify_full() == 0))
for wv in [QQ(1) / 2, 1, 2]:
    delta = atan2(2 * g * wv, w0 ** 2 - wv ** 2)
    print("phase lag at omega =", wv, ":", delta, "=", n(delta))
