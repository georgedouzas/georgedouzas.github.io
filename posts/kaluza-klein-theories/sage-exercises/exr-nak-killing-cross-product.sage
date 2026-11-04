# Exercise exr-nak-killing-cross-product: the Killing vectors of the unit sphere seen in R^3. The push-forward of
# K_a = K_a^m d_m by the embedding n(th, ph) is the rotation field e_a x n; the brackets [K_a, K_b] = -eps_abc K_c
# follow from the cross product, and K_a . K_b = (e_a x n).(e_b x n) = delta_ab - n_a n_b.
th, ph = var("th ph")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
J = matrix(SR, 3, 2, lambda i, m: diff(n[i], [th, ph][m]))          # columns d n / d th, d n / d ph
e = identity_matrix(SR, 3).columns()
push = [(J * K[a]).simplify_full() for a in range(3)]
for a in range(3): print("push-forward of K_%d:" % (a + 1), push[a], "  e_%d x n:" % (a + 1), e[a].cross_product(n))
print("[PASS] the push-forward of K_a is e_a x n:", all((push[a] - e[a].cross_product(n)).simplify_full().is_zero() for a in range(3)))
# linear vector fields on R^3: X_a(x) = e_a x x = L_a x with the matrices (L_a)_{bc} = -eps_abc
x = vector(var("x1 x2 x3"))
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
L = [matrix(QQ, 3, 3, lambda b, c: -lc(a, b, c)) for a in range(3)]
print("[PASS] e_a x x = L_a x with (L_a)_{bc} = -eps_abc:", all((L[a] * x - e[a].cross_product(x)).is_zero() for a in range(3)))
# for linear fields X = L x, Y = M x the vector-field bracket is [X, Y] = (M L - L M) x
br = lambda P, Q: Q * P - P * Q
print("[PASS] [X_a, X_b] = (L_b L_a - L_a L_b) x = -eps_abc X_c:", all(br(L[a], L[b]) == -sum(lc(a, b, c) * L[c] for c in range(3)) for a in range(3) for b in range(3)))
G = matrix(SR, 3, 3, lambda a, b: (e[a].cross_product(n) * e[b].cross_product(n)).simplify_full())
print("[PASS] (e_a x n).(e_b x n) = delta_ab - n_a n_b:", (G - (identity_matrix(3) - n.column() * n.row())).apply_map(lambda t: t.simplify_full()).is_zero())
