# Exercise exr-dq-ground-state-check: G = A r^gamma e^{-lambda r}, F = ((1 - gamma)/(Z alpha)) G, E = m gamma, lambda = m Z alpha solve
# G' - kappa G/r = -(E - V + m) F, F' + kappa F/r = (E - V - m) G for kappa = 1, V = -Z alpha/r; and the normalisation constant A.
r, m, a, A = var("r m a A", domain="positive")     # a = Z alpha < 1
assume(a < 1)
gm = sqrt(1 - a^2); lam = m * a; E = m * gm; V = -a / r; kap = 1
G = A * r^gm * exp(-lam * r); F = (1 - gm) / a * G
eq1 = diff(G, r) - kap * G / r + (E - V + m) * F
eq2 = diff(F, r) + kap * F / r - (E - V - m) * G
print("first equation:", bool((eq1 / G).simplify_full() == 0), "; second equation:", bool((eq2 / G).simplify_full() == 0))
f = (1 - gm) / a
norm = A^2 * (1 + f^2) * gamma(2 * gm + 1) / (2 * lam)^(2 * gm + 1)
print("int (G^2 + F^2) dr = A^2 (1 + f^2) Gamma(2 gamma + 1)/(2 lambda)^(2 gamma + 1); check at a = 1/2, m = 1, A = 1:",
      abs(numerical_integral(((G^2 + F^2).subs(a=1/2, m=1, A=1)), 0, oo)[0] - norm.subs(a=1/2, m=1, A=1).n()) < 1e-8)
print("A^2 = (2 lambda)^(2 gamma + 1)/((1 + f^2) Gamma(2 gamma + 1)), with 1 + f^2 = 2/(1 + gamma):", bool((1 + f^2 - 2 / (1 + gm)).simplify_full() == 0))
