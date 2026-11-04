# Exercise exr-manifold-s3-charts: the three-sphere S^3 in R^4 with its two stereographic charts: the inverse of the
# north chart lands on S^3, the transition map to the south chart is the inversion (u, v, w)/rho^2 in the unit sphere
# of R^3 with Jacobian determinant -1/rho^6, so S^3 is a smooth manifold of dimension 3.
S3 = Manifold(3, "S^3")
U = S3.open_subset("U"); V = S3.open_subset("V")          # S^3 minus the north pole; minus the south pole
S3.declare_union(U, V)
N = U.chart("u v w"); S = V.chart("up:u' vp:v' wp:w'")
u, v, w = N[:]; up, vp, wp = S[:]
rho2 = u**2 + v**2 + w**2
E = [2*u/(1 + rho2), 2*v/(1 + rho2), 2*w/(1 + rho2), (rho2 - 1)/(1 + rho2)]
print("the north chart's inverse lands on S^3, sum of squares = 1:", bool(sum(e**2 for e in E).simplify_full() == 1))
south = [E[k]/(1 + E[3]) for k in range(3)]               # south stereographic coordinates (X1, X2, X3)/(1 + X4)
print("the south coordinates of that point are the inversion (u, v, w)/rho^2:", all(bool((south[k] - [u, v, w][k]/rho2).simplify_full() == 0) for k in range(3)))
tr = N.transition_map(S, (u/rho2, v/rho2, w/rho2), intersection_name="W", restrictions1=rho2 != 0, restrictions2=up**2 + vp**2 + wp**2 != 0)
tr.set_inverse(up/(up**2 + vp**2 + wp**2), vp/(up**2 + vp**2 + wp**2), wp/(up**2 + vp**2 + wp**2))
print("transition map:", tr.display())
print("Jacobian determinant of the transition map:", jacobian(tr(u, v, w), (u, v, w)).det().factor())
print("atlas:", S3.atlas(), " -> S^3 is a manifold of dimension", S3.dim())
print("the point (u, v, w) = (0, 0, 0) is the south pole (0, 0, 0, -1):", [e.subs(u=0, v=0, w=0) for e in E])
