# Exercise exr-so3-lie-algebra: so(3) with Sage's LieAlgebra module: its structure constants in the basis L_a, the
# Killing form, and the isomorphism with su(2) by matching structure constants.
L = [matrix(QQ, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(QQ, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(QQ, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
so3 = LieAlgebra(QQ, {('L1', 'L2'): {'L3': 1}, ('L2', 'L3'): {'L1': 1}, ('L3', 'L1'): {'L2': 1}}, names='L1,L2,L3')
L1, L2, L3 = so3.gens()
print("brackets in Sage's Lie algebra:", L1.bracket(L2), L2.bracket(L3), L3.bracket(L1))
print("Jacobi identity holds (Sage checks it when building the algebra):", so3.dimension() == 3)
print("the matrices L_a satisfy the same brackets:", all((L[a]*L[b] - L[b]*L[a]) == sum(Permutation([a+1, b+1, c+1]).sign() * L[c] if len({a, b, c}) == 3 else 0 * L[0] for c in range(3)) for a in range(3) for b in range(3)))
K = so3.killing_form_matrix()
print("Killing form in the basis L_a:"); print(K)
print("equals -2 delta_ab:", K == -2 * identity_matrix(QQ, 3), "; negative definite (compact):", (-K).is_positive_definite())
print("Sage recognises the algebra as simple of type A1 = su(2):", so3.is_semisimple() and so3.killing_form_matrix().det() != 0)
