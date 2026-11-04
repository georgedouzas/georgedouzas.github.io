# Exercise exr-ren-x-skeleton: series of the Hartree solution and of the two-loop truncation G^-1 = 1 + lam G/2 - lam^2 G^3/6
# against the exact coefficients 1, -1/2, 2/3, -11/8.
var('lam c1 c2 c3')
print("Hartree:", taylor((sqrt(1 + 2 * lam) - 1) / lam, lam, 0, 3))
g = 1 + c1 * lam + c2 * lam^2 + c3 * lam^3
eq = taylor(g * (1 + lam * g / 2 - lam^2 * g^3 / 6) - 1, lam, 0, 3)
sol = solve([eq.coefficient(lam, k) == 0 for k in [1, 2, 3]], [c1, c2, c3], solution_dict=True)[0]
print("two-loop truncation:", g.subs(sol))
print("exact: 1 - lam/2 + 2/3 lam^2 - 11/8 lam^3")
