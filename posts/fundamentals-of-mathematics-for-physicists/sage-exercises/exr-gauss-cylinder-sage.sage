# Exercise exr-gauss-cylinder-sage: the cylinder and the cone, as surfaces in R^3, have induced metrics with vanishing
# Gauss curvature: bent but flat (SageManifolds pullback and ricci_scalar).
from sage.all import *
E3 = Manifold(3, 'R^3', structure='Riemannian'); C = E3.chart('X Y Z')
h = E3.metric('h'); h[0, 0] = 1; h[1, 1] = 1; h[2, 2] = 1
R0, alpha = var('R0 alpha', domain='real'); assume(R0 > 0, 0 < alpha, alpha < pi/2)
# cylinder of radius R0
Cy = Manifold(2, 'Cyl', structure='Riemannian'); zp = Cy.chart(r'z ph:(0,2*pi):\varphi'); z, ph = zp[:]
emb1 = Cy.diff_map(E3, {(zp, C): [R0*cos(ph), R0*sin(ph), z]}, name='E1')
g1 = Cy.metric('g1'); g1.set(emb1.pullback(h))
print("cylinder: induced metric", g1.display(), "; Christoffel symbols all zero:", g1.christoffel_symbols() == 0, "; K = R/2 =", g1.ricci_scalar().expr())
# cone of half-angle alpha, rho = distance from the apex along the surface
Co = Manifold(2, 'Cone', structure='Riemannian'); rp = Co.chart(r'rho:(0,+oo) ph:(0,2*pi):\varphi'); rho, ph2 = rp[:]
emb2 = Co.diff_map(E3, {(rp, C): [rho*sin(alpha)*cos(ph2), rho*sin(alpha)*sin(ph2), rho*cos(alpha)]}, name='E2')
g2 = Co.metric('g2'); g2.set(emb2.pullback(h))
print("cone: induced metric", g2.display())
Gam = g2.christoffel_symbols()
print("cone: Gamma^rho_phph =", Gam[0, 1, 1].expr().simplify_full(), ", Gamma^ph_rhoph =", Gam[1, 0, 1].expr().simplify_full(), "; K = R/2 =", g2.ricci_scalar().expr().simplify_full())
print("both surfaces are flat:", bool(g1.ricci_scalar().expr() == 0) and bool(g2.ricci_scalar().expr().simplify_full() == 0))
