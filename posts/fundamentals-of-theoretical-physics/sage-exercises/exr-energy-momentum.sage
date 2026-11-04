# Exercise exr-energy-momentum: the canonical energy-momentum tensor of a real scalar in 1+1 dimensions,
# T^{mu nu} = d^mu phi d^nu phi - eta^{mu nu} L (signature (+, -)), its components (energy density, momentum density,
# pressure), and its conservation on the Klein-Gordon equation, with Sage.
t, x, m = var("t x m", domain="real")
phi = function("phi")(t, x)
eta = diagonal_matrix(SR, [1, -1]); X = [t, x]
d = [diff(phi, X[mu]) for mu in range(2)]                  # lower-index derivatives
dup = [sum(eta[mu, nu] * d[nu] for nu in range(2)) for mu in range(2)]
L = sum(eta[mu, mu] * d[mu]**2 for mu in range(2)) / 2 - m**2 * phi**2 / 2     # L = (1/2) eta^{mu nu} d phi d phi - (1/2) m^2 phi^2
T = matrix(SR, 2, 2, lambda mu, nu: dup[mu] * dup[nu] - eta[mu, nu] * L)     # T^{mu nu} = d^mu phi d^nu phi - eta^{mu nu} L
print("T^{tt} (energy density) =", T[0, 0].expand())
print("T^{tx} (momentum density) =", T[0, 1].expand(), "; T^{xx} (pressure) =", T[1, 1].expand())
box = diff(phi, t, 2) - diff(phi, x, 2)                    # Box = eta^{mu nu} d_mu d_nu = d_t^2 - d_x^2
for nu in range(2):
    div = sum(diff(T[mu, nu], X[mu]) for mu in range(2)).simplify_full()
    print(f"d_mu T^(mu {X[nu]}) =", div.factor(), " = (Box phi + m^2 phi) d^nu phi:", bool((div - (box + m**2 * phi) * dup[nu]).simplify_full() == 0))
print("on the Klein-Gordon equation (Box + m^2) phi = 0 both divergences vanish: energy and momentum are conserved")
