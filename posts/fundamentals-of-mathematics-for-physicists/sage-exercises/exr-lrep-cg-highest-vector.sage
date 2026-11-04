# Exercise exr-lrep-cg-highest-vector: the singlet in 1 x 1 and the triplet in 1/2 x 1/2 as kernels of the total
# raising operator J_+ x 1 + 1 x J_+ restricted to a weight space.
def spin_matrices(j):
    ms = [j - k for k in range(2 * j + 1)]; n = len(ms)
    J3 = diagonal_matrix(SR, ms)
    Jp = matrix(SR, n, n, lambda r, c: sqrt(j * (j + 1) - ms[c] * (ms[c] + 1)) if r == c - 1 else 0)
    return J3, Jp
for j in [QQ(1)/2, 1]:
    J3, Jp = spin_matrices(j); n = J3.nrows(); One = identity_matrix(SR, n)
    T3 = J3.tensor_product(One) + One.tensor_product(J3); Tp = Jp.tensor_product(One) + One.tensor_product(Jp)
    zero_weight = [k for k in range(n * n) if T3[k, k] == 0]
    M = Tp.matrix_from_columns(zero_weight)
    ker = M.right_kernel().basis()
    names = [f"|{j - a // n}>|{j - a % n}>" for a in zero_weight]
    for v in ker:
        print(f"{j} x {j}: highest-weight vector of weight 0 (killed by J+):", " + ".join(f"({c})" + nm for c, nm in zip(v, names) if c != 0))
    print(f"   number of highest-weight vectors of weight 0 = number of spin-0 summands:", len(ker))
