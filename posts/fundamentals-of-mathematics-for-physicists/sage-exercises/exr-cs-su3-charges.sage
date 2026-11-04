# Exercise exr-cs-su3-charges: the isotropy charges of the flag manifold SU(3)/U(1)xU(1). Check every bracket of
# @prp-coset-su3 with the Gell-Mann matrices, the opposite charges of the conjugate combinations lambda_a - i lambda_b,
# and the non-symmetric bracket [lambda_1, lambda_4] = i lambda_7.
lam = [None] + [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
br = lambda X, Y: X * Y - Y * X
for (a, b) in [(1, 2), (4, 5), (6, 7)]:
    for s in (1, -1):
        v = lam[a] + s * I * lam[b]
        q = []
        for h in (3, 8):
            w = br(lam[h], v); idx = next((i, j) for i in range(3) for j in range(3) if v[i, j] != 0)
            c = (w[idx] / v[idx]).simplify_full()
            assert bool((w - c * v).simplify_full() == 0)
            q.append(c)
        print(f"lambda_{a} {'+' if s == 1 else '-'} i lambda_{b}: charges (ad lambda_3, ad lambda_8) = ({q[0]}, {q[1]})")
print("[lambda_1, lambda_4] = i lambda_7:", bool(br(lam[1], lam[4]) == I * lam[7]))
