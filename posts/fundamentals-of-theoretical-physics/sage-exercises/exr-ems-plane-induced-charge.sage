# Exercise exr-ems-plane-induced-charge (Macchi et al., Problem 2.16): induced charge on a grounded plane.
q, a, s, t = var('q a s t', domain='positive')
sigma = -q * a / (2 * pi * (t^2 + a^2)^(3/2))
Qs = integrate(2 * pi * t * sigma, t, 0, s).simplify_full()
print("charge within radius s:", Qs)
print("total:", limit(Qs, s=oo))
print("half of the charge within s =", solve(Qs == -q / 2, s))
