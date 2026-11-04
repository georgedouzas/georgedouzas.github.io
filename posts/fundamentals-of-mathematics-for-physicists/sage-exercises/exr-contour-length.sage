# Exercise exr-contour-length: lengths of the circle of radius r, the segment from 0 to 1 + i and the parabolic arc
# t + i t^2, 0 <= t <= 1, as integrals of |gamma'(t)|.
t = var("t", domain="real"); r = var("r", domain="positive")
def length(g, a, b):
    gp = diff(g, t)
    return integrate(sqrt((gp.real_part())^2 + (gp.imag_part())^2), t, a, b)
print("circle r e^{it}:", length(r*exp(I*t), 0, 2*pi))
print("segment (1 + i) t:", length((1 + I)*t, 0, 1))
L = length(t + I*t^2, 0, 1)
print("parabola t + i t^2:", L, "=", L.n())
