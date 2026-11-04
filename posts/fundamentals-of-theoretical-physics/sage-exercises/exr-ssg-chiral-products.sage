# Exercise exr-ssg-chiral-products: at a fixed point y the chiral superfields Phi_i = phi_i + sqrt2 theta psi_i + theta theta F_i
# are polynomials in theta^1, theta^2 whose fermion components psi_{i alpha} are Grassmann generators too. We multiply
# two of them and read off the components of Phi_1 Phi_2, then compute the theta theta component of
# W(Phi) = L Phi + M/2 Phi^2 + y/6 Phi^3 for one field and of a cubic superpotential of two fields, and compare with
# W^i F_i - 1/2 W^{ij} psi_i psi_j (psi chi = psi^alpha chi_alpha, eps^{12} = 1).
K.<r2> = QuadraticField(2)
R = PolynomialRing(K, "p1, p2, F1, F2, L, M, y, a, b, c, d"); p1, p2, F1, F2, L, M, y, a, b, c, d = R.gens()
E = ExteriorAlgebra(R, ["t1", "t2", "u1", "u2", "v1", "v2"]); t1, t2, u1, u2, v1, v2 = E.gens()
th = [t1, t2]; psi = [[u1, u2], [v1, v2]]          # psi_{i alpha}, lower index
eps_up = matrix(R, [[0, 1], [-1, 0]])
def dot(x, z):                                      # x^alpha z_alpha = eps^{alpha beta} x_beta z_alpha for lower-index x, z
    return sum(eps_up[al, be] * x[be] * z[al] for al in range(2) for be in range(2))
thth = -2 * t1 * t2                                 # theta theta for upper-index theta^1, theta^2
thpsi = [sum(th[al] * psi[i][al] for al in range(2)) for i in range(2)]   # theta psi = theta^alpha psi_alpha
Phi = [p1 + r2 * thpsi[0] + thth * F1, p2 + r2 * thpsi[1] + thth * F2]
prod12 = Phi[0] * Phi[1]
expected = p1 * p2 + r2 * (p1 * thpsi[1] + p2 * thpsi[0]) + thth * (p1 * F2 + F1 * p2 - dot(psi[0], psi[1]))
print("[PASS] Phi1 Phi2 = phi1 phi2 + sqrt2 theta(phi1 psi2 + psi1 phi2) + theta theta (phi1 F2 + F1 phi2 - psi1 psi2):", prod12 == expected)
# one field: W = L Phi + M/2 Phi^2 + y/6 Phi^3
P = Phi[0]; W = L * P + M / 2 * P^2 + y / 6 * P^3
lhs = W - (L * p1 + M / 2 * p1^2 + y / 6 * p1^3)  # drop the lowest component W(phi)
Wp = L + M * p1 + y / 2 * p1^2; Wpp = M + y * p1
target = thth * (Wp * F1 - Wpp / 2 * dot(psi[0], psi[0])) + r2 * Wp * thpsi[0]
print("[PASS] one field: [W(Phi)]_{theta theta} = W' F - 1/2 W'' psi psi (and the theta component is sqrt2 W' psi):", lhs == target)
# two fields: W = a Phi1 Phi2 + b Phi1^2 Phi2 + c Phi2^3 + d Phi1
W2 = a * Phi[0] * Phi[1] + b * Phi[0]^2 * Phi[1] + c * Phi[1]^3 + d * Phi[0]
Sx = PolynomialRing(R, "x1, x2"); x1, x2 = Sx.gens()
Wf = a * x1 * x2 + b * x1^2 * x2 + c * x2^3 + d * x1
sub = {x1: p1, x2: p2}
Wi = [R(Wf.derivative(x).subs(sub)) for x in (x1, x2)]
Wij = [[R(Wf.derivative(x).derivative(z).subs(sub)) for z in (x1, x2)] for x in (x1, x2)]
top = thth * (Wi[0] * F1 + Wi[1] * F2 - sum(Wij[i][j] * dot(psi[i], psi[j]) for i in range(2) for j in range(2)) / 2)
lin = r2 * (Wi[0] * thpsi[0] + Wi[1] * thpsi[1])
print("[PASS] two fields: [W(Phi)]_{theta theta} = W^i F_i - 1/2 W^{ij} psi_i psi_j:", W2 - R(Wf.subs(sub)) == top + lin)
