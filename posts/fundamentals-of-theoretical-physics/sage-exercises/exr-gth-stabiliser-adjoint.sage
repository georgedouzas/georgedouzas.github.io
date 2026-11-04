# Exercise exr-gth-stabiliser-adjoint: the unbroken subalgebra of an adjoint scalar Phi is the kernel of
# X -> [X, Phi]. For SU(2) with Phi = T_3 it is one-dimensional; for SU(3) with Phi = T_8 (eigenvalues 1, 1, -2 up to a
# factor) it is four-dimensional, su(2) + u(1); for SU(3) with Phi = i diag(1, 2, -3) it is the two-dimensional Cartan
# subalgebra. The number of broken generators is dim G - dim H.
def kernel_dim(T, Phi):
    rows = []
    for X in T:
        C = X * Phi - Phi * X
        rows.append([c.real() for c in C.list()] + [c.imag() for c in C.list()])
    M = matrix(SR, rows)
    return len(T) - M.rank()
s = [matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
T2 = [-I / 2 * m for m in s]
gm = [matrix(QQbar, 3, 3, d) for d in [
    [0, 1, 0, 1, 0, 0, 0, 0, 0], [0, -I, 0, I, 0, 0, 0, 0, 0], [1, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, -I, 0, 0, 0, I, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, -I, 0, I, 0]]] + [matrix(QQbar, 3, 3, [1, 0, 0, 0, 1, 0, 0, 0, -2]) / sqrt(QQbar(3))]
T3 = [-I / 2 * m for m in gm]
print("[PASS] the Gell-Mann generators are normalised, Tr(T_A T_B) = -delta_AB/2:", all((T3[A] * T3[B]).trace() == (-QQbar(1) / 2 if A == B else 0) for A in range(8) for B in range(8)))
k1 = kernel_dim(T2, T2[2]); k2 = kernel_dim(T3, T3[7]); k3 = kernel_dim(T3, I * diagonal_matrix(QQbar, [1, 2, -3]))
print("SU(2), Phi = T_3: dim H =", k1, "; broken generators:", 3 - k1)
print("SU(3), Phi = T_8: dim H =", k2, "; broken generators:", 8 - k2)
print("SU(3), Phi = i diag(1, 2, -3): dim H =", k3, "; broken generators:", 8 - k3)
print("[PASS] SU(2) -> U(1), SU(3) -> SU(2) x U(1), SU(3) -> U(1) x U(1):", (k1, k2, k3) == (1, 4, 2))
print("[PASS] for Phi = T_8 the unbroken generators are T_1, T_2, T_3, T_8:", [A + 1 for A in range(8) if (T3[A] * T3[7] - T3[7] * T3[A]).is_zero()] == [1, 2, 3, 8])
