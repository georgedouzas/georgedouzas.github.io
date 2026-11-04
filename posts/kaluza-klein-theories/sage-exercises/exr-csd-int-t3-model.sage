# Exercise exr-csd-int-t3-model: SU(3) over S^2 with the T_3 embedding tau' = (i/2) diag(1, -1, 0). The constraints
# leave phi_+ = f E_12; the internal field strength F_12 = [phi_1, phi_2] - tau' and the potential -Tr F_12^2.
f, fb = var("f fb")
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
ad = lambda A, B: A * B - B * A
taup = I / 2 * diagonal_matrix(SR, [1, -1, 0])
X = matrix(SR, 3, 3, lambda a, b: var("x%d%d" % (a, b)))
sol = solve([e == 0 for e in (ad(taup, X) - I * X).list()], X.list(), solution_dict=True)[0]
print("[PASS] the +i eigenspace of ad tau' is spanned by E_12:", [v for v in X.list() if sol[v] != 0] == [X[0, 1]])
Pp = f * E(0, 1); Pm = -fb * E(1, 0)
f1 = (Pp + Pm) / 2; f2 = I * (Pp - Pm) / 2
print("[PASS] the singlet solves the constraints [tau', phi_1] = phi_2, [tau', phi_2] = -phi_1:", (ad(taup, f1) - f2).is_zero() and (ad(taup, f2) + f1).is_zero())
F = (ad(f1, f2) - taup).expand(); V = -(F * F).trace().expand()
print("F_12 =", F.diagonal(), "; V =", V.factor())
print("[PASS] V = (|f|^2 - 1)^2 / 2: minimum 0 at |f| = 1, where F_12 = 0 (a flat vacuum, unlike the 1/6 of the lambda_8 model):",
      bool((V - (f * fb - 1)^2 / 2).expand() == 0) and F.subs({f: 1, fb: 1}).is_zero())
