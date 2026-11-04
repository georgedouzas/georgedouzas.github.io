# Exercise exr-grb-ds2-wave (Blennow-Ohlsson, Problem 2.140): the wave operator g^{mu nu} nabla_mu nabla_nu Phi on
# two-dimensional de Sitter space ds^2 = dt^2 - exp(2t/R) dx^2, computed with SageManifolds, its homogeneous
# solutions, and the plane waves in conformal time eta = -R exp(-t/R).
M = Manifold(2, 'M', structure='Lorentzian'); X.<t, x> = M.chart(); R = var('R'); assume(R > 0)
g = M.metric('g'); g[0, 0] = 1; g[1, 1] = -exp(2 * t / R)
Phi = M.scalar_field(function('Phi')(t, x), name='Phi')
nab = g.connection(); H = nab(Phi.differential()); gi = g.inverse()
box = sum(gi[a, b].expr() * H[a, b].expr() for a in range(2) for b in range(2))   # g^{ab} nabla_a nabla_b Phi
print("Box Phi =", box.simplify_full())
expected = diff(function('Phi')(t, x), t, 2) + diff(function('Phi')(t, x), t) / R - exp(-2 * t / R) * diff(function('Phi')(t, x), x, 2)
print("Box Phi = Phi_tt + Phi_t/R - exp(-2t/R) Phi_xx:", bool((box - expected).simplify_full() == 0))
A, B, k, eta = var('A B k eta')
f = A + B * exp(-t / R)
print("homogeneous solutions A + B exp(-t/R):", bool((diff(f, t, 2) + diff(f, t) / R).simplify_full() == 0))
w = exp(I * k * x) * exp(-I * k * (-R * exp(-t / R)))   # exp(i k (x - eta)) with eta = -R exp(-t/R)
print("exp(i k (x - eta)) solves the wave equation:", bool((diff(w, t, 2) + diff(w, t) / R - exp(-2 * t / R) * diff(w, x, 2)).simplify_full() == 0))
