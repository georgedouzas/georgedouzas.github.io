# Exercise exr-kkr-gauge-inverse-metric: the inverse ansatz transforms contravariantly, G'^{-1} = J^{-1} G^{-1} J^{-T},
# and the result is the inverse ansatz with A replaced by A - d eps: g^{mu nu} and phi unchanged, A^mu -> A^mu - d^mu eps.
gi = matrix(SR, 4, 4, lambda i, j: var("gi%d%d" % (min(i, j), max(i, j))))      # g^{mu nu} as free symbols
A = vector(SR, var("A0 A1 A2 A3")); de = vector(SR, var("e0 e1 e2 e3")); phi = var("phi")
def inv_ansatz(A):
    Au = gi * A
    M = matrix(SR, 5, 5)
    for i in range(4):
        for j in range(4): M[i, j] = gi[i, j]
        M[i, 4] = M[4, i] = -Au[i]
    M[4, 4] = -phi^(-2) + A * Au
    return M
J = identity_matrix(SR, 5)
for i in range(4): J[4, i] = -de[i]
Ji = J.inverse()
lhs = Ji * inv_ansatz(A) * Ji.transpose()
print("[PASS] J^{-1} G^{-1}(A) J^{-T} = G^{-1}(A - d eps):", bool((lhs - inv_ansatz(A - de)).expand() == 0))
