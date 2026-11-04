# Exercise exr-acsym-emt-potential: for L = (1/2)(phi_t^2 - phi_x^2) - V(phi) with an arbitrary potential V in 1+1
# dimensions (signature (+, -)), T^{mu nu} = d^mu phi d^nu phi - eta^{mu nu} L satisfies
# d_mu T^{mu nu} = (Box phi + V'(phi)) d^nu phi identically.
t, x, p = var('t x p', domain='real'); X = [t, x]
eta = diagonal_matrix(SR, [1, -1])
phi = function('phi')(t, x)
Vp = function('V')(p)
d = [diff(phi, X[mu]) for mu in range(2)]; dup = [eta[mu, mu] * d[mu] for mu in range(2)]
L = (d[0]^2 - d[1]^2) / 2 - Vp.subs(p=phi)
T = matrix(SR, 2, 2, lambda mu, nu: dup[mu] * dup[nu] - eta[mu, nu] * L)
print("T^{tt} =", T[0, 0].expand())
box = diff(phi, t, 2) - diff(phi, x, 2)
Vprime = diff(Vp, p).subs(p=phi)
for nu in range(2):
    div = sum(diff(T[mu, nu], X[mu]) for mu in range(2))
    print("nu = %s: d_mu T^{mu nu} = (Box phi + V'(phi)) d^nu phi:" % X[nu], bool((div - (box + Vprime) * dup[nu]).simplify_full() == 0))
