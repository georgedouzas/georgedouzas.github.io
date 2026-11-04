# Exercise exr-flag-nonsymmetric: the coset SU(3)/U(1)xU(1) is reductive but not symmetric: with the Gell-Mann
# matrices, every bracket of a Cartan generator with a coset generator stays in the coset directions, but
# [lambda_1, lambda_4] = i lambda_7 lies in the coset directions too, so [k, k] is not contained in r.
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
r_idx, k_idx = [2, 7], [0, 1, 3, 4, 5, 6]
def coeffs(X):                                           # coefficients of X in the basis lambda_A: c_A = tr(X lambda_A)/2
    return [((X * lam[A]).trace() / 2).simplify_full() for A in range(8)]
def in_span(X, idx):
    c = coeffs(X); return all(c[A] == 0 for A in range(8) if A not in idx)
br = lambda A, B: lam[A] * lam[B] - lam[B] * lam[A]
print("[r, r] in r:", all(in_span(br(i, j), r_idx) for i in r_idx for j in r_idx))
print("[r, k] in k (reductive):", all(in_span(br(i, a), k_idx) for i in r_idx for a in k_idx))
print("[k, k] in r (symmetric)?", all(in_span(br(a, b), r_idx) for a in k_idx for b in k_idx))
print("[lambda_1, lambda_4] =", "i lambda_7:", bool(br(0, 3) == I * lam[6]), " -> a coset direction: the coset is not symmetric")
print("for the sphere SO(3)/SO(2), by contrast, [L_1, L_2] = L_3 lies in r: symmetric")
