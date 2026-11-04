# Exercise exr-sr-divergence-scalar: the divergence of A(x) = (t^2, x y, z, t x) is a Lorentz scalar field under a boost
# with v = 3/5 along x: d'_mu A'^mu(x') equals d_mu A^mu(x) at corresponding points.
var('t x y z tp xp yp zp')
X = vector([t, x, y, z]); A = vector([t^2, x*y, z, t*x])
divA = sum(diff(A[m], X[m]) for m in range(4))
v = 3/5; g = 5/4
Lam = matrix(QQ, [[g, -g*v, 0, 0], [-g*v, g, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
Xp = vector([tp, xp, yp, zp]); Xold = Lam.inverse()*Xp          # x = Lambda^-1 x'
sub = dict(zip(X, Xold))
Ap = Lam*vector([c.subs(sub) for c in A])                        # A'(x') = Lambda A(Lambda^-1 x')
divAp = sum(diff(Ap[m], Xp[m]) for m in range(4))
print("div A =", divA)
print("div' A' at x' minus div A at x = Lambda^-1 x':", (divAp - divA.subs(sub)).simplify_full())
