# Exercise exr-tv-great-circle: the velocity of the great circle E(t) = (cos t, sin t/sqrt2, sin t/sqrt2) on the sphere
# at t = pi/4, in the spherical chart, and its action on the height Z compared with dZ/dt along the curve.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
t = var("t")
Ecurve = vector([cos(t), sin(t)/sqrt(2), sin(t)/sqrt(2)])
th_t, ph_t = arccos(Ecurve[2]), arctan(tan(t)/sqrt(2))      # the chart coordinates of the curve, valid for 0 < t < pi/2
t0 = pi/4
p = (th_t.subs(t=t0).simplify_full(), ph_t.subs(t=t0).simplify_full())
c = (diff(th_t, t).subs(t=t0).simplify_full(), diff(ph_t, t).subs(t=t0).simplify_full())
print("the point (theta, phi) at t = pi/4:", p)
print("the velocity components (theta', phi') at t = pi/4:", c)
X = A.vector_field(c[0], c[1], name="X")
Z = A.scalar_field(cos(th), name="Z")
print("X(Z) at the point:", X(Z).expr().subs(th=p[0]).simplify_full(), "; dZ/dt along the curve:", diff(Ecurve[2], t).subs(t=t0))
E = vector([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)])
amb = (c[0]*E.diff(th) + c[1]*E.diff(ph)).subs(th=p[0], ph=p[1])
print("ambient velocity c^i d_i E equals E'(pi/4):", bool(all((amb[k] - Ecurve.diff(t).subs(t=t0)[k]).simplify_full() == 0 for k in range(3))))
