# Exercise exr-spd-sphere-omega: the spin connection of the unit sphere from the Christoffel symbols,
# omega_{ab mu} = e_a^nu (d_mu e_{b nu} - Gamma^lambda_{mu nu} e_{b lambda}), and the structure equations.
th, ph = var("th ph")
X = [th, ph]
g = diagonal_matrix(SR, [1, sin(th) ^ 2]); gi = g.inverse()
Gam = [[[sum(gi[l, s] * (diff(g[s, m], X[n]) + diff(g[s, n], X[m]) - diff(g[m, n], X[s])) / 2 for s in range(2)).simplify_full()
         for n in range(2)] for m in range(2)] for l in range(2)]
E = matrix(SR, [[1, 0], [0, 1 / sin(th)]])          # E[a, mu] = e_a^mu
Elow = (E * g)                                       # e_{a nu} = e_a^mu g_{mu nu}
def omega(a, b, mu):
    return sum(E[a, nu] * (diff(Elow[b, nu], X[mu]) - sum(Gam[l][mu][nu] * Elow[b, l] for l in range(2))) for nu in range(2)).simplify_full()
for a in range(2):
    for b in range(2):
        print("omega_{%d%d} = (%s) dtheta + (%s) dphi" % (a + 1, b + 1, omega(a, b, 0), omega(a, b, 1)))
# structure equations d e^a + omega^a_b ^ e^b = 0 with coframe e^1 = dtheta, e^2 = sin(theta) dphi:
# the dtheta ^ dphi component of omega^a_b ^ e^b is omega_ab,theta e^b_phi - omega_ab,phi e^b_theta
cof = matrix(SR, [[1, 0], [0, sin(th)]])             # cof[a, mu] = e^a_mu
de = [0, diff(sin(th), th)]                          # d e^a = de[a] dtheta ^ dphi
res = [(de[a] + sum(omega(a, b, 0) * cof[b, 1] - omega(a, b, 1) * cof[b, 0] for b in range(2))).simplify_full() for a in range(2)]
print("d e^a + omega^a_b ^ e^b (coefficient of dtheta ^ dphi):", res)
