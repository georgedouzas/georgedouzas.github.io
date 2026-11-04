# Exercise exr-sd-sage-4x4: the 4 x 4 Hermitian tridiagonal matrix with 1 on the diagonal, i above and -i below: eigenvalues
# 1 + 2 cos(k pi/5) in terms of the golden ratio, the unitary of eigenvectors, A = U D U^dagger, and tr e^{iA}.
A = matrix(SR, 4, 4, lambda j, k: 1 if j == k else (I if k == j + 1 else (-I if k == j - 1 else 0)))
print("A ="); print(A); print("Hermitian:", A.is_hermitian())
D, V = A.eigenmatrix_right()
ev = [e.simplify_full() for e in D.diagonal()]
phi = (1 + sqrt(5)) / 2
print("eigenvalues:", ev)
same = lambda L1, L2: all(abs(a - b) < 1e-12 for a, b in zip(sorted(L1), sorted(L2)))
print("equal to {1 + phi, 1 + 1/phi, 1 - 1/phi, 1 - phi}:", same([e.n() for e in ev], [(1 + phi).n(), (1 + 1 / phi).n(), (1 - 1 / phi).n(), (1 - phi).n()]), "; = 1 + 2 cos(k pi/5):", same([e.n() for e in ev], [(1 + 2 * cos(k * pi / 5)).n() for k in range(1, 5)]))
U = matrix(SR, [c / c.norm() for c in V.columns()]).transpose().simplify_full()
print("U^dagger U = 1:", bool((U.conjugate_transpose() * U).simplify_full() == identity_matrix(4)), "; A = U D U^dagger:", bool((U * D * U.conjugate_transpose()).simplify_full() == A))
E = (U * diagonal_matrix([exp(I * d) for d in D.diagonal()]) * U.conjugate_transpose())
print("tr e^{iA} =", E.trace().n(digits=10), "; sum e^{i lambda_k} =", sum(exp(I * e) for e in ev).n(digits=10), "; via Sage's exp:", (I * A).exp().trace().n(digits=10))
