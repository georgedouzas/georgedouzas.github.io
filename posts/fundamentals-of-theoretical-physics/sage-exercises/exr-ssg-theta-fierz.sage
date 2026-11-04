# Exercise exr-ssg-theta-fierz: the superspace coordinates theta^alpha, thetadagger^alphadot are four Grassmann generators.
# With eps^{12} = 1, eps_{12} = -1 (the conventions of def-weyl-spinor), theta theta = theta^alpha theta_alpha and
# thetadagger thetadagger = thetadagger_alphadot thetadagger^alphadot. We check
#   theta^alpha theta^beta = -1/2 eps^{alpha beta} theta theta,   thetadagger^alphadot thetadagger^betadot = 1/2 eps^{..} thetadagger thetadagger,
#   (theta sigma^mu thetadagger)(theta sigma^nu thetadagger) = 1/2 eta^{mu nu} theta theta thetadagger thetadagger  (signature (+,-,-,-)),
# and the expansion phi(x + i theta sigma thetadagger) = phi + i theta sigma^mu thetadagger d_mu phi - 1/4 theta theta thetadagger thetadagger box phi.
K.<I> = QuadraticField(-1)
E = ExteriorAlgebra(K, ["th1", "th2", "tb1", "tb2"]); th1, th2, tb1, tb2 = E.gens()
th = [th1, th2]; tb = [tb1, tb2]
eps_up = matrix(K, [[0, 1], [-1, 0]]); eps_dn = matrix(K, [[0, -1], [1, 0]])
thl = [sum(eps_dn[a, b] * th[b] for b in range(2)) for a in range(2)]
tbl = [sum(eps_dn[a, b] * tb[b] for b in range(2)) for a in range(2)]
thth = sum(th[a] * thl[a] for a in range(2)); tbtb = sum(tbl[a] * tb[a] for a in range(2))
print("theta theta =", thth, "   thetadagger thetadagger =", tbtb)
print("[PASS] theta^a theta^b = -1/2 eps^{ab} theta theta:", all(th[a] * th[b] == -eps_up[a, b] / 2 * thth for a in range(2) for b in range(2)))
print("[PASS] thetadagger^a thetadagger^b = +1/2 eps^{ab} thetadagger thetadagger:", all(tb[a] * tb[b] == eps_up[a, b] / 2 * tbtb for a in range(2) for b in range(2)))
sig = [identity_matrix(K, 2), matrix(K, [[0, 1], [1, 0]]), matrix(K, [[0, -I], [I, 0]]), matrix(K, [[1, 0], [0, -1]])]
eta = diagonal_matrix(K, [1, -1, -1, -1])
V = [sum(th[a] * sig[m][a, b] * tb[b] for a in range(2) for b in range(2)) for m in range(4)]
print("[PASS] (theta sigma^mu thetadagger)(theta sigma^nu thetadagger) = 1/2 eta^{mu nu} theta theta thetadagger thetadagger, all 16 pairs:",
      all(V[m] * V[n] == eta[m, n] / 2 * thth * tbtb for m in range(4) for n in range(4)))
# Taylor expansion of phi(y): the second-order term is (1/2) (i theta sigma^mu thetadagger)(i theta sigma^nu thetadagger) d_mu d_nu phi
R = PolynomialRing(K, ["d%d%d" % (m, n) for m in range(4) for n in range(4)]); dd = matrix(R, 4, 4, R.gens())
ER = ExteriorAlgebra(R, ["th1", "th2", "tb1", "tb2"]); g = ER.gens()
VR = [sum(g[a] * sig[m][a, b] * g[2 + b] for a in range(2) for b in range(2)) for m in range(4)]
second = sum((I * VR[m]) * (I * VR[n]) * dd[m, n] for m in range(4) for n in range(4)) / 2
box = sum(eta[m, m] * dd[m, m] for m in range(4))   # d_mu d^mu with symmetric second derivatives
ththR = -2 * g[0] * g[1]; tbtbR = 2 * g[2] * g[3]
symm = {dd[m, n]: dd[n, m] for m in range(4) for n in range(m)}
diff = second + ththR * tbtbR * box / 4
print("[PASS] the second-order term is -1/4 theta theta thetadagger thetadagger box phi (box = d_mu d^mu):",
      all(c.subs(symm) == 0 for c in diff.monomial_coefficients().values()))
# the fermion term of the expansion: (theta sigma^mu thetadagger)(theta chi) = -1/2 theta theta (chi sigma^mu thetadagger)
E3 = ExteriorAlgebra(K, ["th1", "th2", "tb1", "tb2", "c1", "c2"]); h = E3.gens()
chi = [h[4], h[5]]; chi_up = [sum(eps_up[a, b] * chi[b] for b in range(2)) for a in range(2)]
ok = all(sum(h[a] * sig[m][a, b] * h[2 + b] for a in range(2) for b in range(2)) * sum(h[a] * chi[a] for a in range(2))
         == -QQ(1) / 2 * (-2 * h[0] * h[1]) * sum(chi_up[a] * sig[m][a, b] * h[2 + b] for a in range(2) for b in range(2)) for m in range(4))
print("[PASS] (theta sigma^mu thetadagger)(theta chi) = -1/2 theta theta (chi sigma^mu thetadagger) for a Grassmann spinor chi:", ok)
