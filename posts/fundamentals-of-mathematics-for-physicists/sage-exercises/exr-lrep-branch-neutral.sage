# Exercise exr-lrep-branch-neutral: the brackets behind 8 -> 3_0 + 1_0 + 2_{sqrt3} + 2_{-sqrt3}: lambda_8 commutes with
# lambda_1, lambda_2, lambda_3, which close into su(2), and acts on lambda_4 + i lambda_5, lambda_6 + i lambda_7 by sqrt(3).
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
br = lambda A, B: A * B - B * A
l8 = lam[7]
print("[lambda_8, lambda_a] = 0 for a = 1, 2, 3:", all(bool(br(l8, lam[a]) == 0) for a in range(3)))
print("[lambda_1, lambda_2] = 2i lambda_3, cyclically:", bool(br(lam[0], lam[1]) == 2*I*lam[2]), bool(br(lam[1], lam[2]) == 2*I*lam[0]), bool(br(lam[2], lam[0]) == 2*I*lam[1]))
for name, E in [("lambda_4 + i lambda_5", lam[3] + I*lam[4]), ("lambda_6 + i lambda_7", lam[5] + I*lam[6]),
                ("lambda_4 - i lambda_5", lam[3] - I*lam[4]), ("lambda_6 - i lambda_7", lam[5] - I*lam[6])]:
    q8 = [ (br(l8, E)[i, j] / E[i, j]).simplify_full() for i in range(3) for j in range(3) if E[i, j] != 0][0]
    q3 = [ (br(lam[2], E)[i, j] / E[i, j]).simplify_full() for i in range(3) for j in range(3) if E[i, j] != 0][0]
    print(f"{name}: lambda_8-charge {q8}, lambda_3-weight {q3}")
