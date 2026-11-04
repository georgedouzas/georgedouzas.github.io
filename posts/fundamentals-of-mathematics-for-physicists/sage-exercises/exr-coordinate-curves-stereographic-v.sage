# Exercise exr-coordinate-curves-stereographic-v: the coordinate curves of the north stereographic chart are
# circles through the north pole: the v-curve u = c lies on the plane X = c (1 - Z) through N, the u-curve v = c on
# the plane Y = c (1 - Z), and the circle cut out by such a plane has radius 1/sqrt(1 + c^2).
u, v, c = var("u v c", domain="real")
E = vector([2*u/(1 + u**2 + v**2), 2*v/(1 + u**2 + v**2), (u**2 + v**2 - 1)/(1 + u**2 + v**2)])
Ev = E.subs(u=c)                                   # the v-curve u = c, parametrised by v
print("the v-curve u = c lies on the plane X = c (1 - Z):", bool((Ev[0] - c*(1 - Ev[2])).simplify_full() == 0))
print("the plane contains N = (0, 0, 1):", bool(0 == c*(1 - 1)))
Eu = E.subs(v=c)                                   # the u-curve v = c, parametrised by u
print("the u-curve v = c lies on the plane Y = c (1 - Z):", bool((Eu[1] - c*(1 - Eu[2])).simplify_full() == 0))
# the plane X + c Z = c has unit normal (1, 0, c)/sqrt(1 + c^2) and distance |c|/sqrt(1 + c^2) from the centre;
# it cuts the unit sphere in a circle of radius sqrt(1 - distance^2)
d = abs(c)/sqrt(1 + c**2)
print("radius of the circle:", sqrt(1 - d**2).simplify_full())
print("for c = 0 the v-curve is the great circle X = 0 (a meridian), radius", sqrt(1 - d**2).subs(c=0))
print("the v-curve approaches N at both ends, v -> +-oo:", (limit(Ev[0], v=oo), limit(Ev[1], v=oo), limit(Ev[2], v=oo)))
