# Exercise exr-cs-integrative-s3: S^3 = SO(4)/SO(3) from end to end. The representative
# L = exp(phi M_01) exp(theta M_12) exp(chi M_23) sends o = e_3 to the point with hyperspherical coordinates
# (chi, theta, phi); the components of L^{-1} dL along k = span(M_03, M_13, M_23) are the coset vielbein, their sum of
# squares is the round metric, its Ricci scalar is 6, and the r-components form the SO(3)-connection.
ch, th, ph = var("ch th ph", domain="real")
def M(i, j):
    m = matrix(SR, 4, 4); m[i, j] = 1; m[j, i] = -1; return m
simp = lambda A: A.apply_map(lambda x: x.simplify_full())
def R(i, j, t):                                   # exp(t M_ij): a rotation in the (i, j)-plane
    m = identity_matrix(SR, 4); m[i, i] = m[j, j] = cos(t); m[i, j] = sin(t); m[j, i] = -sin(t); return m
print("exp(t M_12) agrees with the matrix exponential:", simp((ch * M(1, 2)).exp().apply_map(lambda x: x.rectform())) == R(1, 2, ch))
L = simp(R(0, 1, ph) * R(1, 2, th) * R(2, 3, ch))
o = vector(SR, [0, 0, 0, 1])
print("L . o =", (L * o).apply_map(lambda x: x.simplify_full()).list())
coords = [ch, th, ph]
MC = [simp(L.transpose() * diff(L, x)) for x in coords]            # L^{-1} = L^T for a rotation
kk = [(0, 3), (1, 3), (2, 3)]; rr = [(0, 1), (0, 2), (1, 2)]
e = [[MC[m][p] for m in range(3)] for p in kk]                    # coefficient of M_ij is the (i, j) entry
w = [[MC[m][p] for m in range(3)] for p in rr]
for (p, comp) in zip(kk, e):
    print(f"e along M_{p[0]}{p[1]}: ", [c.simplify_full() for c in comp], " (d chi, d theta, d phi)")
for (p, comp) in zip(rr, w):
    print(f"r-component along M_{p[0]}{p[1]}: ", [c.simplify_full() for c in comp])
G = matrix(SR, 3, 3, lambda m, n: sum(e[a][m] * e[a][n] for a in range(3)).simplify_full())
print("sum_a e^a e^a =", G.rows())
S = Manifold(3, "S3", structure="Riemannian")
Y.<chi, theta, phi> = S.chart(r"chi:(0,pi):\chi theta:(0,pi):\theta phi:(0,2*pi):\phi")
g = S.metric(); g[0, 0], g[1, 1], g[2, 2] = 1, sin(chi)^2, sin(chi)^2 * sin(theta)^2
print("the round metric:", G == diagonal_matrix([1, sin(ch)^2, sin(ch)^2 * sin(th)^2]), " Ricci scalar:", g.ricci_scalar().expr())
