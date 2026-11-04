# Exercise exr-gth-ym-cubic-quartic: for constant SU(2) potentials (no derivatives), F^a_{mu nu} = g eps_{abc} A^b_mu A^c_nu
# and the Yang-Mills Lagrangian -1/4 F^a F^a equals the quartic self-interaction
# -(g^2/4) [ (A_mu . A^mu)^2 - (A_mu . A_nu)(A^mu . A^nu) ], with "." the sum over the internal index a, in (+, -, -, -).
g = var("g")
Asym = [[var("A%d_%d" % (a, m)) for m in range(4)] for a in range(3)]
eta = diagonal_matrix([1, -1, -1, -1])
eps = lambda a, b, c: (a - b) * (b - c) * (c - a) / 2
F = [[[g * sum(eps(a, b, c) * Asym[b][m] * Asym[c][n] for b in range(3) for c in range(3)) for n in range(4)] for m in range(4)] for a in range(3)]
lag = -sum(eta[m, m] * eta[n, n] * F[a][m][n]^2 for a in range(3) for m in range(4) for n in range(4)) / 4
dot = lambda m, n: sum(Asym[a][m] * Asym[a][n] for a in range(3))          # A_m . A_n (internal)
AA = sum(eta[m, m] * dot(m, m) for m in range(4))                             # A_mu . A^mu
cross = sum(eta[m, m] * eta[n, n] * dot(m, n)^2 for m in range(4) for n in range(4))
print("[PASS] -1/4 F^a F^a = -(g^2/4)[(A.A)^2 - (A_mu.A_nu)(A^mu.A^nu)] for constant fields:", bool((lag + g^2 / 4 * (AA^2 - cross)).expand() == 0))
print("[PASS] the quartic term is proportional to g^2 (homogeneous of degree 2 in g):", bool((lag.diff(g, 2) * g^2 / 2 - lag).expand() == 0))
Ab = {Asym[a][m]: 0 for a in range(3) for m in range(4)}; Ab.update({Asym[0][1]: var("c1"), Asym[1][2]: var("c2")})
print("on A^1_1 = c1, A^2_2 = c2 the Lagrangian is", lag.subs(Ab).expand(), "= -(g^2/2) (F^3_12)^2 with F^3_12 = g c1 c2")
