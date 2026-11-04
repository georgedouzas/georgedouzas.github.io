# Exercise exr-lrep-spinj-matrices: the spin-j matrices in the weight basis |j, m>, m = j, ..., -j:
# J_3 = diag(m), J_+|m> = sqrt(j(j+1) - m(m+1)) |m+1>, J_- = J_+^T; check the brackets and the Casimir.
def spin_matrices(j):
    ms = [j - k for k in range(2 * j + 1)]
    n = len(ms)
    J3 = diagonal_matrix(SR, ms)
    Jp = matrix(SR, n, n, lambda r, c: sqrt(j * (j + 1) - ms[c] * (ms[c] + 1)) if r == c - 1 else 0)
    return J3, Jp, Jp.transpose()
br = lambda A, B: A * B - B * A
for j in [QQ(1)/2, 1, QQ(3)/2, 2, QQ(5)/2, 3]:
    J3, Jp, Jm = spin_matrices(j)
    cas = (Jm * Jp + J3^2 + J3).simplify_full()
    ok = bool(br(J3, Jp) == Jp) and bool(br(J3, Jm) == -Jm) and bool((br(Jp, Jm) - 2 * J3).simplify_full() == 0)
    print(f"j = {j}: dimension {J3.nrows()}, brackets hold: {ok}, Casimir J-J+ + J3^2 + J3 = j(j+1) 1: {bool(cas == j * (j + 1) * identity_matrix(J3.nrows()))}")
J3, Jp, Jm = spin_matrices(QQ(3)/2)
print("spin 3/2, J_+ ="); print(Jp)
