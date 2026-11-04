# Exercise exr-gamma-duplication: Legendre's duplication formula Gamma(z) Gamma(z + 1/2) = 2^{1 - 2z} sqrt(pi) Gamma(2z),
# exactly at z = 1, 2, 3, 7/2 and numerically at z = 0.3 + 0.7 i; and Gamma(1/4) Gamma(3/4) = pi sqrt(2).
for z0 in (1, 2, 3, 7/2):
    lhs = gamma(z0)*gamma(z0 + 1/2); rhs = 2^(1 - 2*z0)*sqrt(pi)*gamma(2*z0)
    print("z =", z0, ":", lhs, "==", rhs.simplify_full(), ":", bool((lhs - rhs).simplify_full() == 0))
z0 = 0.3 + 0.7*I
print("z = 0.3 + 0.7 i (numerically):", abs((gamma(z0)*gamma(z0 + 1/2) - 2^(1 - 2*z0)*sqrt(pi)*gamma(2*z0)).n()) < 1e-12)
print("Gamma(1/4) Gamma(3/4) by duplication at z = 1/4: 2^(1/2) sqrt(pi) Gamma(1/2) =", (2^(1/2)*sqrt(pi)*gamma(1/2)).simplify_full(), "  numerically", (gamma(1/4)*gamma(3/4)).n(), (pi*sqrt(2)).n())
