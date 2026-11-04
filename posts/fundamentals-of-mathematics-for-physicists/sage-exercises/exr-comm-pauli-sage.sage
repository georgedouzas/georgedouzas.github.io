# Exercise exr-comm-pauli-sage: the commutators and anticommutators of the Pauli matrices, the product formula
# sigma_a sigma_b = delta_ab + i eps_abc sigma_c, and the Jacobi identity on two triples.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
comm = lambda A, B: A * B - B * A
acomm = lambda A, B: A * B + B * A
eps = lambda a, b, c: Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
for a in range(3):
    for b in range(3):
        print("[s%d, s%d] =" % (a + 1, b + 1), comm(s[a], s[b]).list(), " {s%d, s%d} =" % (a + 1, b + 1), acomm(s[a], s[b]).list())
ok = all(bool(s[a] * s[b] == (1 if a == b else 0) * identity_matrix(2) + I * sum(eps(a, b, c) * s[c] for c in range(3))) for a in range(3) for b in range(3))
print("sigma_a sigma_b = delta_ab + i eps_abc sigma_c:", ok)
jac = lambda A, B, C: comm(A, comm(B, C)) + comm(B, comm(C, A)) + comm(C, comm(A, B))
print("Jacobi for (s1, s2, s3):", jac(s[0], s[1], s[2]).is_zero(), "; for (s1, s1 s2, s3):", jac(s[0], s[0] * s[1], s[2]).is_zero())
