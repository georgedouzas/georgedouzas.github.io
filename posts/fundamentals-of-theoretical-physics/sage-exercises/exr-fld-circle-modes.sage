# Exercise exr-fld-circle-modes: the Fourier modes phi_n = (1/(2 pi R)) int_0^{2 pi R} phi e^{-i n y/R} dy of fields
# on a circle of radius R: cos(y/R)^3 has the modes n = +-1 and +-3 only, and the modes rebuild the field;
# sin(y/R)^2 + cos(2y/R) has a constant mode and n = +-2.
y = var("y", domain="real"); R = var("R"); assume(R > 0)
def mode(f, n):
    return (integrate(f * exp(-I * n * y / R), y, 0, 2 * pi * R) / (2 * pi * R)).simplify_full()
f = cos(y / R)**3
print("modes of cos(y/R)^3, n = -4..4:", [mode(f, n) for n in range(-4, 5)])
recon = sum(mode(f, n) * exp(I * n * y / R) for n in range(-3, 4))
print("the four non-zero modes rebuild the field:", bool((recon.rectform() - f).trig_reduce().simplify_full() == 0))
g = sin(y / R)**2 + cos(2 * y / R)
print("modes of sin(y/R)^2 + cos(2y/R), n = -3..3:", [mode(g, n) for n in range(-3, 4)])
