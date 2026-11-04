# Exercise exr-emm-gi-hamiltonian: the transformed Hamiltonian is the minimally coupled one of the new potentials.
var('m q t')
X0, X1, X2, P0, P1, P2 = var('X0 X1 X2 P0 P1 P2'); Xv = [X0, X1, X2]; Pv = [P0, P1, P2]
phi = function('phi')(t, X0, X1, X2); A = [function('A%d' % i)(t, X0, X1, X2) for i in range(3)]; lam = function('lam')(t, X0, X1, X2)
Hf = lambda p, ph, AA: sum((p[i] - q * AA[i])^2 for i in range(3)) / (2 * m) + q * ph
old_p = [Pv[i] - q * diff(lam, Xv[i]) for i in range(3)]          # p = P - q grad(lambda)
K = Hf(old_p, phi, A) - q * diff(lam, t)
Hnew = Hf(Pv, phi - diff(lam, t), [A[i] + diff(lam, Xv[i]) for i in range(3)])
print('K - H_new =', (K - Hnew).simplify_full())
