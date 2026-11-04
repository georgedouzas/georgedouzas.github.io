# Exercise exr-noether-current: the U(1) current of a complex scalar and its divergence, signature (+, -, -, -).
t, x, y, z, m = var('t x y z m', domain='real')
X = [t, x, y, z]; eta = [1, -1, -1, -1]
u = function('u')(t, x, y, z); v = function('v')(t, x, y, z)          # phi = u + i v
phi = u + I*v; phibar = u - I*v
L = sum(eta[mu]*diff(phibar, X[mu])*diff(phi, X[mu]) for mu in range(4)) - m^2*phibar*phi
# Noether: delta phi = i phi, J^mu = dL/d(d_mu phi) delta phi + dL/d(d_mu phibar) delta phibar
J = [(eta[mu]*diff(phibar, X[mu]))*(I*phi) + (eta[mu]*diff(phi, X[mu]))*(-I*phibar) for mu in range(4)]
J = [j.expand() for j in J]
print("Noether current J^0 =", J[0])
j = [-Jm for Jm in J]                                                 # the conventional sign: j^mu = i(phibar d^mu phi - phi d^mu phibar)
print("j^mu = -J^mu = i(phibar d^mu phi - phi d^mu phibar):", all(bool((j[mu] - I*eta[mu]*(phibar*diff(phi, X[mu]) - phi*diff(phibar, X[mu]))).expand() == 0) for mu in range(4)))
div = sum(diff(j[mu], X[mu]) for mu in range(4)).expand()
box = lambda f: sum(eta[mu]*diff(f, X[mu], 2) for mu in range(4))    # Box = d_t^2 - nabla^2
print("div j = i(phibar box phi - phi box phibar):", bool((div - I*(phibar*box(phi) - phi*box(phibar))).expand() == 0))
# on-shell: (Box + m^2) phi = 0, i.e. box phi = -m^2 phi, so div j = -i m^2 (phibar phi - phi phibar) = 0
print("on-shell divergence:", (I*(phibar*(-m^2*phi) - phi*(-m^2*phibar))).expand())
