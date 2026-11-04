# Exercise exr-ems-exponential-density: rho = rho0 exp(-r/a); enclosed charge, field, total charge, limits.
rho0, a, r, t = var('rho0 a r t', domain='positive')
Qr = (4 * pi * rho0 * integrate(t^2 * exp(-t / a), t, 0, r)).simplify_full()
print("Q(r) =", Qr)
print("total charge:", limit(Qr, r=oo))
E = Qr / (4 * pi * r^2)
print("E(r) for small r:", E.taylor(r, 0, 2))
print("r^2 E(r) at large r ->", limit(r^2 * E, r=oo), "= Q/(4 pi)")
