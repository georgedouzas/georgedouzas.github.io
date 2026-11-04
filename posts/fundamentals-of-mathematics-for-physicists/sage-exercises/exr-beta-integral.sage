# Exercise exr-beta-integral: B(3, 4) as an integral and as Gamma(3) Gamma(4)/Gamma(7); the Wallis integral of
# sin^4 as a Beta function.
t = var("t", domain="real")
print("integral of t^2 (1 - t)^3 over [0, 1]:", integrate(t^2*(1 - t)^3, t, 0, 1))
print("beta(3, 4) =", beta(3, 4), "  Gamma(3) Gamma(4)/Gamma(7) =", gamma(3)*gamma(4)/gamma(7))
w = integrate(sin(t)^4, t, 0, pi/2)
print("integral of sin^4 over [0, pi/2]:", w, "  (1/2) B(5/2, 1/2) =", (beta(5/2, 1/2)/2).simplify_full(), "  equal:", bool((w - beta(5/2, 1/2)/2).simplify_full() == 0))
