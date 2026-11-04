# Exercise exr-fan-dual-norm-lagrange: maximise x1 + 4 x2 on x1^3 + x2^3 = 1, x >= 0, with a Lagrange multiplier.
x1, x2, mu = var('x1 x2 mu')
sols = solve([1 == 3 * mu * x1^2, 4 == 3 * mu * x2^2, x1^3 + x2^3 == 1], x1, x2, mu, solution_dict=True)
real = [s for s in sols if all(s[v] in RR and s[v] > 0 for v in (x1, x2))]
for s in real:
    print("maximiser:", s[x1], s[x2], " ratio x2/x1 =", (s[x2] / s[x1]).simplify_full(), " value:", (s[x1] + 4 * s[x2]).n(digits=8))
print("|(1, 4)|_{3/2} = 9^(2/3) =", (9^(2 / 3)).n(digits=8))
