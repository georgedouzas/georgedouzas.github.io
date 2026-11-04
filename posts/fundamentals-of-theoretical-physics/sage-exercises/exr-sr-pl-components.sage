# Exercise exr-sr-pl-components: the Pauli-Lubanski vector W_mu = (1/2) eps_{mu nu rho sigma} M^{nu rho} k^sigma for a rest
# momentum and a light-like momentum, with formal antisymmetric M^{nu rho}; J_i = (1/2) eps_ijk M^{jk}, K_i = M^{0i}.
var('m kappa J1 J2 J3 K1 K2 K3')
Mf = matrix(SR, 4, 4)
Mf[0, 1], Mf[0, 2], Mf[0, 3] = K1, K2, K3
Mf[2, 3], Mf[3, 1], Mf[1, 2] = J1, J2, J3
Mf = Mf - Mf.transpose()
def eps4(*idx):
    return 0 if len(set(idx)) < 4 else sign(Permutation([i + 1 for i in idx]).signature())
def W(k):
    return [sum(eps4(mu, nu, rho, sg)*Mf[nu, rho]*k[sg] for nu in range(4) for rho in range(4) for sg in range(4))/2 for mu in range(4)]
print("rest momentum (m,0,0,0):    W_mu =", W([m, 0, 0, 0]))
print("light-like (kappa,0,0,kappa): W_mu =", [w.expand() for w in W([kappa, 0, 0, kappa])])
