# Exercise exr-su3-structure: the structure constants of su(3) in the Gell-Mann basis, split for the coset
# SU(3)/U(1)xU(1): r = span(lambda_3, lambda_8), k = the other six. f_ab^c with a, b, c all in k are non-zero.
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
T = [-I/2*l for l in lam]                                   # anti-Hermitian generators with [T_A, T_B] = f_AB^C T_C
def coeffs(X):                                               # X = sum c_C T_C  <=>  c_C = -2 Tr(T_C X)
    return [(-2*(T[C]*X).trace()).simplify_full() for C in range(8)]
f = {}
for A in range(8):
    for B in range(A+1, 8):
        for C, c in enumerate(coeffs(T[A]*T[B] - T[B]*T[A])):
            if c != 0: f[(A+1, B+1, C+1)] = c
print("non-zero f_AB^C (A < B):", f)
r = {3, 8}; k = {1, 2, 4, 5, 6, 7}
print("reductive ([r, k] in k):", all(C in k for (A, B, C) in f if (A in r) != (B in r)))
print("f_ab^c with a, b, c in k (non-symmetric):", {key: v for key, v in f.items() if set(key) <= k})
print("f_ab^i with a, b in k, i in r:", {key: v for key, v in f.items() if key[0] in k and key[1] in k and key[2] in r})
