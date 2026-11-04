# Exercise exr-oraifeartaigh: the O'Raifeartaigh model W = -k Phi_1 + m Phi_2 Phi_3 + (y/2) Phi_1 Phi_3^2 breaks
# supersymmetry (Martin, hep-ph/9709356, Sec. 7.3): compute the F-terms, show that they cannot vanish together,
# find the minimum and the tree-level spectrum at the origin, and check the supertrace sum rule.
k, m, y = var("k m y"); assume(k > 0, m > 0, y > 0, m^2 > y * k)
p1, p2, p3 = var("p1 p2 p3"); c1, c2, c3 = var("c1 c2 c3")        # fields and their conjugates as independent symbols
W = -k * p1 + m * p2 * p3 + y / 2 * p1 * p3^2
Wi = [diff(W, v) for v in (p1, p2, p3)]
Fs = [-w for w in Wi]                                                 # F_i^* = -W^i
print("F-terms (conjugated): F_1 =", -Wi[0].subs(p3=c3), " F_2 =", -Wi[1].subs(p3=c3), " F_3 =", -Wi[2].subs(p1=c1, p3=c3))
print("[PASS] F_2 = 0 forces phi_3 = 0, and then F_1 = k != 0: no supersymmetric vacuum:", solve(Wi[1] == 0, p3) == [p3 == 0] and bool(Wi[0].subs(p3=0) == -k))
V = sum(w * w.subs({p1: c1, p2: c2, p3: c3}) for w in Wi)             # V = sum |W^i|^2
print("[PASS] V at the origin is k^2 > 0:", bool(V.subs({p1: 0, p2: 0, p3: 0, c1: 0, c2: 0, c3: 0}) == k^2))
# tree-level mass matrix of the six real scalars at the origin, with phi_j = (x_j + i w_j)/sqrt(2)
xs = var("x1 x2 x3 w1 w2 w3")
sub = {p1: (xs[0] + I * xs[3]) / sqrt(2), p2: (xs[1] + I * xs[4]) / sqrt(2), p3: (xs[2] + I * xs[5]) / sqrt(2),
       c1: (xs[0] - I * xs[3]) / sqrt(2), c2: (xs[1] - I * xs[4]) / sqrt(2), c3: (xs[2] - I * xs[5]) / sqrt(2)}
Vr = V.subs(sub).expand()
M2 = matrix(SR, 6, 6, lambda i, j: diff(Vr, xs[i], xs[j])).subs({v: 0 for v in xs})
scalars = sorted([e.simplify_full() for e in M2.eigenvalues()], key=str)
print("scalar mass-squared eigenvalues at the origin:", scalars)
print("[PASS] the scalar spectrum is 0, 0, m^2, m^2, m^2 - yk, m^2 + yk (Martin eq. 7.3.9):",
      sorted(scalars, key=str) == sorted([0, 0, m^2, m^2, m^2 - k * y, m^2 + k * y], key=str))
# fermion mass matrix W^{ij} at the origin and its squared eigenvalues
Wij = matrix(SR, 3, 3, lambda i, j: diff(W, (p1, p2, p3)[i], (p1, p2, p3)[j])).subs({p1: 0, p2: 0, p3: 0})
fermions = sorted([e for e in (Wij.conjugate_transpose() * Wij).eigenvalues()], key=str)
print("fermion mass-squared eigenvalues:", fermions)
print("[PASS] the fermion spectrum is 0, m^2, m^2 (eq. 7.3.10): the massless one is the goldstino:", sorted(fermions, key=str) == sorted([0, m^2, m^2], key=str))
print("[PASS] the supertrace sum rule STr(m^2) = sum(scalars) - 2 sum(fermions) = 0 (eq. 7.1.13):", bool((sum(scalars) - 2 * sum(fermions)).simplify_full() == 0))
