# Exercise exr-kkr-cylinder-fourier: why the zero modes form a consistent truncation. On the circle y ~ y + 2 pi the
# n-th Fourier coefficient of a product of two modes e^{i a y} e^{i b y} is nonzero only for n = a + b; so products of
# zero modes have only a zero mode, while the product of the modes 1 and -1 has a zero-mode component: massive modes
# source the massless ones, but not conversely.
y = var("y")
def coeff(f, n):
    return integrate(f * exp(-I * n * y), y, 0, 2 * pi) / (2 * pi)
f0 = 3; g0 = 5                                                  # two y-independent fields
print("[PASS] the product of two zero modes has only the n = 0 coefficient:",
      bool(coeff(f0 * g0, 0) == 15) and all(bool(coeff(f0 * g0, n) == 0) for n in (1, 2, -1)))
p = exp(I * y) * exp(-I * y)
print("coefficients of e^{iy} e^{-iy} for n = -2..2:", [coeff(p, n) for n in range(-2, 3)])
print("[PASS] the modes n = 1 and n = -1 together source the zero mode:", bool(coeff(p, 0) == 1))
q = exp(I * y) * exp(I * y)
print("[PASS] e^{iy} e^{iy} has only the n = 2 coefficient (momentum is conserved):",
      bool(coeff(q, 2) == 1) and all(bool(coeff(q, n) == 0) for n in (0, 1, -2)))
