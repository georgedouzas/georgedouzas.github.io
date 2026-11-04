# Exercise exr-nak-ansatz-components: the components of the non-abelian ansatz on M_4 x S^2 read off from the line
# element g_{mu nu} dx dx - gamma_{mn} (dy^m + K^m_a A^a_mu dx^mu)(dy^n + K^n_b A^b_nu dx^nu), here with a
# two-dimensional spacetime (t, x) for brevity, compared with the formulas of @exm-ansatz-sphere-components; and the
# horizontal vectors d_mu - B^m_mu d_m are orthogonal to the fibre and have ghat(e_mu, e_nu) = g_{mu nu}.
th, ph = var("th ph")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
gam = diagonal_matrix(SR, [1, sin(th)^2])
g = matrix(SR, [[var("g00"), var("g01")], [var("g01"), var("g11")]])
A = matrix(SR, 3, 2, lambda a, mu: var("A%d%d" % (a + 1, mu)))
B = matrix(SR, 2, 2, lambda m, mu: sum(K[a][m] * A[a, mu] for a in range(3)))   # B^m_mu = K^m_a A^a_mu
dX = vector(var("dt dx dth dph"))
theta = vector([dX[2 + m] + sum(B[m, mu] * dX[mu] for mu in range(2)) for m in range(2)])
ds2 = (vector(dX[:2]) * g * vector(dX[:2]) - theta * gam * theta).expand()
G = matrix(SR, 4, 4, lambda i, j: (diff(ds2, dX[i], dX[j]) / 2).simplify_full())   # ds^2 = G_ij dX^i dX^j
print("ghat_{t th} =", G[0, 2].trig_reduce(), "   ghat_{t ph} =", G[0, 3].simplify_trig())
print("[PASS] ghat_{mu th} = sin(ph) A^1_mu - cos(ph) A^2_mu:", all(bool((G[mu, 2] - (sin(ph) * A[0, mu] - cos(ph) * A[1, mu])).simplify_full() == 0) for mu in range(2)))
print("[PASS] ghat_{mu ph} = sin th cos th (cos ph A^1 + sin ph A^2) - sin^2 th A^3:", all(bool((G[mu, 3] - (sin(th) * cos(th) * (cos(ph) * A[0, mu] + sin(ph) * A[1, mu]) - sin(th)^2 * A[2, mu])).simplify_full() == 0) for mu in range(2)))
print("[PASS] ghat_{mn} = -gamma_{mn} and ghat_{mu nu} = g_{mu nu} - gamma_{mn} B^m_mu B^n_nu:", (G[2:, 2:] + gam).is_zero() and (G[:2, :2] - (g - B.transpose() * gam * B)).apply_map(lambda t: t.simplify_full()).is_zero())
E = [vector([1 if i == mu else 0 for i in range(2)] + [-B[0, mu], -B[1, mu]]) for mu in range(2)]   # horizontal e_mu
V = [vector([0, 0, 1, 0]), vector([0, 0, 0, 1])]                                                       # vertical d_m
print("[PASS] the horizontal vectors e_mu = d_mu - B^m_mu d_m are orthogonal to the fibre:", all(bool((E[mu] * G * V[m]).simplify_full() == 0) for mu in range(2) for m in range(2)))
print("[PASS] ghat(e_mu, e_nu) = g_{mu nu}:", all(bool((E[mu] * G * E[nu] - g[mu, nu]).simplify_full() == 0) for mu in range(2) for nu in range(2)))
