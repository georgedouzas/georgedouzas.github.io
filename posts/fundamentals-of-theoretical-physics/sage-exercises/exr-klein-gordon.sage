# Exercise exr-klein-gordon: Euler-Lagrange equation of the real scalar field from its Lagrangian density.
# The Lagrangian is a function of the jet variables p (the field) and p_mu (its gradient); the Euler-Lagrange
# expression is d_mu (dL/dp_mu) - dL/dp with the field phi(x) substituted. Signature (+, -, -, -).
t, x, y, z, m = var('t x y z m', domain='real')
p = var('p'); dp = var('p_t p_x p_y p_z')                    # jet variables
X = [t, x, y, z]; eta = [1, -1, -1, -1]                     # signature (+, -, -, -)
L = sum(eta[mu]*dp[mu]^2 for mu in range(4))/2 - m^2*p^2/2  # L = (1/2) eta^{mu nu} d_mu phi d_nu phi - (1/2) m^2 phi^2
phi = function('phi')(t, x, y, z)
subs = {p: phi, **{dp[mu]: diff(phi, X[mu]) for mu in range(4)}}
EL = sum(diff(diff(L, dp[mu]).subs(subs), X[mu]) for mu in range(4)) - diff(L, p).subs(subs)
print("Euler-Lagrange expression:", EL.expand())
box = sum(eta[mu]*diff(phi, X[mu], 2) for mu in range(4))   # the d'Alembertian d_t^2 - nabla^2 with (+,-,-,-)
print("equals (box phi + m^2 phi), the Klein-Gordon equation (Box + m^2) phi = 0:", bool((EL - (box + m^2*phi)).expand() == 0))
