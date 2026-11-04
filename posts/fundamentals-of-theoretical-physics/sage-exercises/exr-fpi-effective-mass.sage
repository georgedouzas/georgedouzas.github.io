# Exercise exr-fpi-effective-mass: on a periodic time lattice of T = 24 sites the zero-momentum correlator of a free
# field with m a = 1/2 is C(t) = cosh(E(t - T/2)) up to a constant, a E = 2 arcsinh(1/4). The effective mass
# log(C(t)/C(t+1)) equals E only for t << T/2, while the cosh mass arccosh((C(t-1) + C(t+1))/(2 C(t))) equals E for all t.
R = RealField(60)
T, M = 24, R(1) / 2
E = 2 * arcsinh(M / 2)
C = lambda t: sum(cos(2 * R.pi() * nu * t / T) / (M^2 + 4 * sin(R.pi() * nu / T)^2) for nu in range(T)) / T
print("a E = 2 arcsinh(m a/2) =", E.n(digits=6), "; continuum m a =", M)
print(" t   C(t)            log(C(t)/C(t+1))   cosh mass")
for t in range(1, 13):
    c0, c1, cm = C(t), C(t + 1), C(t - 1)
    print(f"{t:2d}   {c0.n(digits=8):<14}  {log(c0 / c1).n(digits=6):<17}  {arccosh((cm + c1) / (2 * c0)).n(digits=6)}")
print("check at t = 3: log(cosh(E(T/2 - t))/cosh(E(T/2 - t - 1))) =",
      log(cosh(E * (T / 2 - 3)) / cosh(E * (T / 2 - 4))).n(digits=6))
