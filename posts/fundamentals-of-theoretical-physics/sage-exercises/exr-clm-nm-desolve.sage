# Exercise exr-clm-nm-desolve: two coupled oscillators (m = k = 1, kappa = 1/5) solved with desolve_system and
# compared with the beat.
t = var("t")
x1, x2, v1, v2 = [function(n)(t) for n in ("x1", "x2", "v1", "v2")]
kap = 1 / 5
# first-order form: x' = v, v' = -M^{-1} K x
eqs = [diff(x1, t) == v1, diff(x2, t) == v2, diff(v1, t) == -(1 + kap) * x1 + kap * x2, diff(v2, t) == kap * x1 - (1 + kap) * x2]
sol = desolve_system(eqs, [x1, x2, v1, v2], ics=[0, 1, 0, 0, 0])[:2]
for s in sol: print(s)
w1, w2 = 1, sqrt(1 + 2 * kap)
beat = [(cos(w1 * t) + cos(w2 * t)) / 2, (cos(w1 * t) - cos(w2 * t)) / 2]
print("agrees with the beat:", all(bool((s.rhs() - b).canonicalize_radical() == 0) for s, b in zip(sol, beat)))
