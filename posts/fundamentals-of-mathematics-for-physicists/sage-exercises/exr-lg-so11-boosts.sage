# Exercise exr-lg-so11-boosts: the boosts B(w) of the Minkowski plane form a one-parameter group, B(v) B(w) = B(v + w),
# preserve eta = diag(1, -1), B^T eta B = eta, have det 1, and are not periodic (trace 2 cosh w > 2 for w != 0).
v, w = var("v w", domain="real")
def B(u): return matrix(SR, [[cosh(u), sinh(u)], [sinh(u), cosh(u)]])
eta = diagonal_matrix(SR, [1, -1])
print("B(v) B(w) - B(v + w) =", (B(v) * B(w) - B(v + w)).apply_map(lambda e: e.exponentialize().simplify_full()).list())
print("B^T eta B - eta =", (B(w).transpose() * eta * B(w) - eta).apply_map(lambda e: e.exponentialize().simplify_full()).list())
print("det B =", B(w).det().exponentialize().simplify_full(), "; B B^T = 1?", bool((B(w) * B(w).transpose()).simplify_full() == 1), "(not orthogonal)")
print("tr B(w) = 2 cosh w, equal to 2 only at w = 0:", solve(2 * cosh(w) == 2, w))
