# Exercise exr-dispersion: plane-wave solutions of the Klein-Gordon equation (Box + m^2) phi = 0 in 3+1 dimensions
# with Sage, signature (+, -, -, -) so that Box = d_t^2 - nabla^2: the dispersion relation omega^2 = k^2 + m^2, the
# phase and group velocities, and the massless limit.
t, x, y, z, m, w = var("t x y z m w", domain="real"); assume(m > 0)
k1, k2, k3 = var("k1 k2 k3", domain="real")
phi = cos(k1 * x + k2 * y + k3 * z - w * t)
box = diff(phi, t, 2) - diff(phi, x, 2) - diff(phi, y, 2) - diff(phi, z, 2)       # Box = d_t^2 - nabla^2
KG = box + m**2 * phi
print("(Box + m^2) phi / phi =", (KG / phi).simplify_full())
wsol = solve(w**2 == k1**2 + k2**2 + k3**2 + m**2, w)
print("dispersion relation:", wsol)
k = var("k", domain="real"); assume(k > 0); wk = sqrt(k**2 + m**2)
print("phase velocity omega/k =", (wk / k).simplify_full(), " > 1 ; group velocity d omega/dk =", diff(wk, k).simplify_full(), " < 1 (the particle velocity)")
print("product of the two velocities:", ((wk / k) * diff(wk, k)).simplify_full(), " ; massless limit: both tend to", limit(diff(wk, k), m=0))
