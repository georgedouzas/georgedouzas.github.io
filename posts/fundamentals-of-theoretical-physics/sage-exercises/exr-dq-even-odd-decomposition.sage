# Exercise exr-dq-even-odd-decomposition: even part (A + beta A beta)/2 and odd part (A - beta A beta)/2 in the Dirac representation
# for gamma^1, gamma_5, gamma^0 gamma_5, Sigma_3 and H = alpha_3 p + beta m + V.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
beta = blk(one, Z, Z, -one); gk = [blk(Z, sk, -sk, Z) for sk in s]
g5 = I * beta * gk[0] * gk[1] * gk[2]
p, m, V = var("p m V")
mats = {"gamma^1": gk[0], "gamma_5": g5, "gamma^0 gamma_5": beta * g5, "Sigma_3": blk(s[2], Z, Z, s[2]), "H": p * beta * gk[2] + m * beta + V * identity_matrix(SR, 4)}
diag_part = lambda M: M.submatrix(0, 0, 2, 2).is_zero() and M.submatrix(2, 2, 2, 2).is_zero()
offd_part = lambda M: M.submatrix(0, 2, 2, 2).is_zero() and M.submatrix(2, 0, 2, 2).is_zero()
for name, A in mats.items():
    ev = (A + beta * A * beta) / 2; od = (A - beta * A * beta) / 2
    print(name, ": even part block diagonal", offd_part(ev), ", odd part block off-diagonal", diag_part(od), "; even part zero:", ev.is_zero(), ", odd part zero:", od.is_zero())
print("odd part of H =", ((mats["H"] - beta * mats["H"] * beta) / 2).list())
