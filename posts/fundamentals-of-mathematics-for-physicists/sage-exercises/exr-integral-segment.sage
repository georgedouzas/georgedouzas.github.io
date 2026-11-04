# Exercise exr-integral-segment: integrals of z and of conj(z) from 0 to 1 + i along the segment and along the
# parabola t + i t^2: path independent for z, path dependent for conj(z).
t = var("t", domain="real")
paths = {"segment": (1 + I)*t, "parabola": t + I*t^2}
for name, g in paths.items():
    gp = diff(g, t)
    Iz = integrate(g*gp, t, 0, 1)
    Izb = integrate(conjugate(g)*gp, t, 0, 1)
    print("%-8s: integral of z dz = %s,  integral of conj(z) dz = %s" % (name, Iz.simplify_full(), Izb.simplify_full()))
print("primitive check for z: (1 + i)^2/2 =", (SR(1 + I)^2/2).simplify_full())
