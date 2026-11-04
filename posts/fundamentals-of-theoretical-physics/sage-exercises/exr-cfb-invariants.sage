# Exercise exr-cfb-invariants (Maggiore, Sec. 8.5): transformation of E and B under a boost along x (signature (+,-,-,-),
# F_{0i} = E_i, F_{ij} = -eps_{ijk} B_k, c = 1), the invariants F_{mu nu}F^{mu nu} = 2(B^2 - E^2) and E.B, and the frame
# moving with v = E x B / B^2 in which crossed fields with |E| < |B| become purely magnetic.
v = var('v'); assume(-1 < v, v < 1); g = 1 / sqrt(1 - v^2)
E = vector(SR, var('E1 E2 E3')); B = vector(SR, var('B1 B2 B3'))
def F_low(E, B):
    F = matrix(SR, 4, 4)
    for i in range(3):
        F[0, i + 1] = E[i]; F[i + 1, 0] = -E[i]
    eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1]).signature()) if len({i, j, k}) == 3 else 0
    for i in range(3):
        for j in range(3):
            F[i + 1, j + 1] = -sum(eps(i, j, k) * B[k] for k in range(3))
    return F
eta = diagonal_matrix(SR, [1, -1, -1, -1])
L = matrix(SR, [[g, -g * v, 0, 0], [-g * v, g, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   # x'^mu = L^mu_nu x^nu
Linv = L.inverse()
Fp = (Linv.transpose() * F_low(E, B) * Linv).simplify_full()       # F'_{mu nu} = (L^-1)^a_mu (L^-1)^b_nu F_ab
Ep = vector([Fp[0, i + 1] for i in range(3)]); Bp = vector([-Fp[2, 3], Fp[1, 3], -Fp[1, 2]])
gam = var('gamma')
tog = lambda e: (e * sqrt(1 - v^2)).simplify_full() * gam      # write the factor 1/sqrt(1 - v^2) as gamma
print("E' =", [Ep[0]] + [tog(e) for e in Ep[1:]])
print("B' =", [Bp[0]] + [tog(b) for b in Bp[1:]])
inv = lambda F: (F * eta * F.transpose() * eta).trace()           # F_{mu nu} F^{mu nu} = -tr(F eta F eta) up to sign; computed directly below
FF = lambda E, B: sum(F_low(E, B)[a, b] * (eta * F_low(E, B) * eta)[a, b] for a in range(4) for b in range(4))
print("F.F = 2(B^2 - E^2):", bool((FF(E, B) - 2 * (B * B - E * E)).simplify_full() == 0))
print("B'^2 - E'^2 = B^2 - E^2:", bool(((Bp * Bp - Ep * Ep) - (B * B - E * E)).simplify_full() == 0))
print("E'.B' = E.B:", bool((Ep * Bp - E * B).simplify_full() == 0))
Ey, Bz = var('Ey Bz'); assume(Bz > 0, Ey > 0, Ey < Bz)
Ec = vector(SR, [0, Ey, 0]); Bc = vector(SR, [0, 0, Bz])
Fc = (Linv.transpose() * F_low(Ec, Bc) * Linv).subs(v=Ey / Bz).simplify_full()
print("crossed fields in the frame v = E/B: E' =", [Fc[0, i + 1] for i in range(3)], " B'_z =", (-Fc[1, 2]).simplify_full())
