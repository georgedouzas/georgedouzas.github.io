# Exercise exr-height-charts: the height Z as a smooth function on the sphere, expressed in the spherical chart and
# in the north stereographic chart with SageManifolds' scalar fields, and checked to agree through the transition map.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A"); U = S2.open_subset("U")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); ste = U.chart("u v")
th, ph = sph[:]; u, v = ste[:]
tr = sph.transition_map(ste, (sin(th)*cos(ph)/(1 - cos(th)), sin(th)*sin(ph)/(1 - cos(th))))
tr.set_inverse(2*atan(1/sqrt(u**2 + v**2)), atan2(v, u), check=False)
W = A.intersection(U)
Z = W.scalar_field({sph.restrict(W): cos(th)}, name="Z")
print("Z in spherical coordinates:", Z.expr(sph.restrict(W)))
print("Z in stereographic coordinates (through the transition map):", Z.expr(ste.restrict(W)).simplify_full())
print("agrees with (u^2 + v^2 - 1)/(u^2 + v^2 + 1):", bool((Z.expr(ste.restrict(W)) - (u**2 + v**2 - 1)/(u**2 + v**2 + 1)).simplify_full() == 0))
print("the angle phi itself has no single-valued expression: atan2(v, u) jumps by 2*pi across the half-line v = 0, u < 0:", bool(atan2(0, -1) - limit(atan2(x, -1), x=0, dir="-") == 2*pi) if False else bool(atan2(0, -1) == pi))
