# Exercise exr-kkr-radius-solve: solve the normalisation e = sqrt(2) kappa/R for R, express it through alpha and M_P
# (kappa^2 = 8 pi/M_P^2, e^2 = 4 pi alpha): R = 2/(sqrt(alpha) M_P), and evaluate it.
e, kappa, R, alpha, MP = var("e kappa R alpha MP"); assume(R > 0, kappa > 0, e > 0, alpha > 0, MP > 0)
sol = solve(e == sqrt(2) * kappa / R, R)[0].rhs()
print("R =", sol)
Rexpr = sol.subs(kappa=sqrt(8 * pi) / MP, e=sqrt(4 * pi * alpha)).simplify_full()
print("[PASS] R = sqrt(2) kappa/e = 2/(sqrt(alpha) M_P):", bool(sol == sqrt(2) * kappa / e) and bool((Rexpr^2 - 4 / (alpha * MP^2)).simplify_full() == 0))
val = Rexpr.subs(alpha=1/137.036, MP=1)
print("R M_P = %.3f for alpha = 1/137.036" % val.n())
print("[PASS] R is about 23.4 Planck lengths:", 23.3 < val.n() < 23.5)
