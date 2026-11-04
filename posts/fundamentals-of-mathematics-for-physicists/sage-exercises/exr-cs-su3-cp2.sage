# Exercise exr-cs-su3-cp2: the complex projective plane CP^2 = SU(3)/U(2), with r = span(lambda_1, lambda_2,
# lambda_3, lambda_8) (the block-diagonal u(2)) and k = span(lambda_4, lambda_5, lambda_6, lambda_7): check that r is a
# subalgebra, that the decomposition is reductive and symmetric, and that ad lambda_8 gives every coset direction the
# same charge, in contrast with the flag manifold.
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
coeffs = lambda X: [((X * lam[A]).trace() / 2).simplify_full() for A in range(8)]
in_span = lambda X, idx: all(c == 0 for A, c in enumerate(coeffs(X)) if A not in idx)
br = lambda A, B: lam[A] * lam[B] - lam[B] * lam[A]
r, k = [0, 1, 2, 7], [3, 4, 5, 6]
print("[r, r] in r (u(2) is a subalgebra):", all(in_span(br(i, j), r) for i in r for j in r))
print("[r, k] in k (reductive):", all(in_span(br(i, a), k) for i in r for a in k))
print("[k, k] in r (symmetric):", all(in_span(br(a, b), r) for a in k for b in k))
for (a, b) in [(3, 4), (5, 6)]:
    v = lam[a] + I * lam[b]; w = lam[7] * v - v * lam[7]
    idx = next((i, j) for i in range(3) for j in range(3) if v[i, j] != 0)
    print(f"[lambda_8, lambda_{a+1} + i lambda_{b+1}] = {(w[idx] / v[idx]).simplify_full()} (lambda_{a+1} + i lambda_{b+1})")
print("dim CP^2 = 8 - 4 =", len(k))
