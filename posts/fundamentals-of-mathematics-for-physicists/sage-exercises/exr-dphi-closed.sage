# Exercise exr-dphi-closed: the one-form dphi on the circle: its integral around the circle is 2*pi, so it is not the
# differential of a function on the circle, although its exterior derivative vanishes (SageManifolds).
S1 = Manifold(1, "S^1"); U = S1.open_subset("U")
c = U.chart("ph:(0,2*pi)"); ph = c[0]
om = U.one_form(1, name="omega")                      # omega = dphi
print("omega =", om.display())
print("d(omega) =", om.exterior_derivative().display(), " (a one-form on a 1-manifold is always closed)")
print("integral of dphi over (0, 2*pi):", integrate(om[0].expr(), ph, 0, 2*pi))
f = U.scalar_field(ph, name="f")
print("locally omega = d(phi):", f.differential() == om, "; but phi is not a function on the whole circle: it jumps by", (2*pi).n(digits=5), "at the point the chart misses")
