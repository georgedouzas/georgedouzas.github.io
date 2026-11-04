# Exercise exr-gamma-recursion: values of the Gamma function and the recursion Gamma(z + 1) = z Gamma(z) at z = 7/3.
print("Gamma(6) =", gamma(6))
print("Gamma(1/2) =", gamma(1/2))
print("Gamma(-3/2) =", gamma(-3/2), "  equals (4/3) sqrt(pi):", bool(gamma(-3/2) == 4*sqrt(pi)/3))
p = gamma(1/3)*gamma(2/3)
print("Gamma(1/3) Gamma(2/3) =", p, "= pi/sin(pi/3) numerically:", abs(p.n() - (pi/sin(pi/3)).n()) < 1e-12, " value", (2*pi/sqrt(3)).n())
z0 = 7/3
print("Gamma(z + 1) == z Gamma(z) at z = 7/3:", bool((gamma(z0 + 1) - z0*gamma(z0)).simplify_full() == 0))
