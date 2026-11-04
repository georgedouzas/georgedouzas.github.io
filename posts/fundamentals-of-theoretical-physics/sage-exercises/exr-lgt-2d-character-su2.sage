# Exercise exr-lgt-2d-character-su2: the character expansion exp(beta cos psi) = (2/beta) sum_j (2j+1) I_{2j+1}(beta) chi_j(psi).
chi = lambda j, p: sin((2*j + 1) * p) / sin(p)
for b, p in ((1.0, 0.7), (3.0, 2.1), (6.0, 1.3)):
    lhs = exp(b * cos(p))
    rhs = sum(2 / b * (2*j + 1) * bessel_I(2*j + 1, b) * chi(j, p) for j in [k / 2 for k in range(0, 41)])
    print('beta = %.1f, psi = %.1f: exp(beta cos psi) = %.10f, character sum = %.10f' % (b, p, lhs, rhs))
b = 2.0
c = lambda j: 2 / b * (2*j + 1) * bessel_I(2*j + 1, b)
print('u = c_(1/2) / (2 c_0) = %.8f, I_2/I_1 = %.8f' % (c(1/2) / (2 * c(0)), bessel_I(2, b) / bessel_I(1, b)))
