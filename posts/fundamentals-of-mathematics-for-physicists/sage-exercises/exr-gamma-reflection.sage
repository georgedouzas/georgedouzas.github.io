# Exercise exr-gamma-reflection: the reflection formula Gamma(z) Gamma(1 - z) = pi/sin(pi z) checked numerically at
# three points, and Gamma(1/2)^2 = pi exactly.
for z0 in (3/10, 1/2 + 2*I, -17/10):
    lhs = (gamma(z0)*gamma(1 - z0)).n(digits=20); rhs = (pi/sin(pi*z0)).n(digits=20)
    print("z =", z0, ": Gamma(z) Gamma(1 - z) =", lhs, "  pi/sin(pi z) =", rhs, "  agree:", abs(lhs - rhs) < 1e-15)
print("Gamma(1/2)^2 =", gamma(1/2)^2)
