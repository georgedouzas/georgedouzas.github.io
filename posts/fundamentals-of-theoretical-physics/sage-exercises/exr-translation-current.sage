# Exercise exr-translation-current: Noether's theorem for spacetime translations of a real scalar in 1+1 dimensions
# (signature (+, -)): the translation along x^nu changes L by the total derivative d_nu L, its Noether current
# J^mu_nu = (dL/d(d_mu phi)) d_nu phi - delta^mu_nu L is the canonical energy-momentum tensor T^mu_nu, and it is
# conserved on the Klein-Gordon equation. Checked symbolically with Sage.
t, x, m = var("t x m", domain="real")
phi = function("phi")(t, x); X = [t, x]; eta = diagonal_matrix(SR, [1, -1])
d = [diff(phi, X[mu]) for mu in range(2)]
L = sum(eta[mu, mu] * d[mu]**2 for mu in range(2)) / 2 - m**2 * phi**2 / 2
dLdd = [eta[mu, mu] * d[mu] for mu in range(2)]                    # dL/d(d_mu phi) = d^mu phi
box = diff(phi, t, 2) - diff(phi, x, 2)                            # Box = d_t^2 - d_x^2
for nu in range(2):
    J = [dLdd[mu] * d[nu] - (L if mu == nu else 0) for mu in range(2)]       # J^mu_nu
    Jup = [eta[nu, nu] * J[mu] for mu in range(2)]                            # J^{mu nu}
    T = [eta[mu, mu] * d[mu] * eta[nu, nu] * d[nu] - eta[mu, nu] * L for mu in range(2)]   # T^{mu nu} = d^mu phi d^nu phi - eta^{mu nu} L
    print(f"translation along {X[nu]}: J^mu_nu =", [j.expand() for j in J])
    print("   J^{mu nu} is the energy-momentum tensor T^{mu nu} (T^{tt} the energy density):", all(bool((Jup[mu] - T[mu]).simplify_full() == 0) for mu in range(2)))
    div = sum(diff(J[mu], X[mu]) for mu in range(2)).simplify_full()
    print("   d_mu J^mu_nu = (Box phi + m^2 phi) d_nu phi, vanishing on the field equation:", bool((div - (box + m**2 * phi) * d[nu]).simplify_full() == 0))
