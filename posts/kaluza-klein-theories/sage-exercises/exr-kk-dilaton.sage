# Exercise exr-kk-dilaton: the dilaton term. For eta - phi(x)^2 dy^2 (signature (+,-,-,-,-)) compute the
# five-dimensional curvature scalar with SageManifolds, compare with -(2/phi) Box phi (Box = d_t^2 - nabla^2, so
# Box phi = -phi''), and show that the reduced action's dilaton term is a total derivative:
# sqrt|g5| R5 = phi (2 phi''/phi) = 2 phi'' = d/dx(2 phi'). Note det g5 = -phi^2 det g4 = phi^2 > 0 here: with four
# spacelike directions the five-dimensional determinant is positive, and the volume element is sqrt|g5| = phi sqrt(-g4).
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x y z w"); t, x, y, z, w = X[:]
phi = function("phi")(x)
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = -1; g[2, 2] = -1; g[3, 3] = -1; g[4, 4] = -phi^2
R5 = g.ricci_scalar().expr().simplify_full()
print("R5 =", R5)
print("[PASS] R5 = 2 phi''/phi = -(2/phi) Box phi with Box phi = -phi'':", bool((R5 - 2 * diff(phi, x, 2) / phi).simplify_full() == 0))
det5 = g.determinant().expr()
print("det g5 =", det5.simplify_full(), "(positive: four spacelike directions)")
print("[PASS] det g5 = phi^2 (so sqrt|g5| = phi for phi > 0):", bool((det5 - phi^2).simplify_full() == 0))
dens = (phi * R5).simplify_full()                       # sqrt|g5| = phi, the circumference factor of the volume element
print("sqrt|g5| R5 = phi R5 =", dens)
print("[PASS] sqrt|g5| R5 = 2 phi'' is a total x-derivative, so the dilaton term drops out of the action:", bool((dens - diff(2 * diff(phi, x), x)).simplify_full() == 0))
