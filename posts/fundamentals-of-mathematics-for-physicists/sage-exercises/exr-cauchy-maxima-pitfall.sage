# Exercise exr-cauchy-maxima-pitfall: the symbolic integral of the parametrised 1/(z - 3) over |z| = 1 returns
# -2 pi i (a branch error of the primitive log(e^{it} - 3)); the correct value 0 is obtained numerically and from
# the Taylor series of 1/(z - 3) on |z| < 3, which has no 1/z term.
t = var("t", domain="real"); z = var("z")
sym = integrate(I*exp(I*t)/(exp(I*t) - 3), t, 0, 2*pi)
print("symbolic integrate:", sym, " (wrong)")
re_, im_ = numerical_integral(lambda s: (I*exp(I*s)/(exp(I*s) - 3)).real(), 0, 2*pi)[0], numerical_integral(lambda s: (I*exp(I*s)/(exp(I*s) - 3)).imag(), 0, 2*pi)[0]
print("numerical integral:", CC(re_, im_), " (correct: 0)")
print("Taylor series of 1/(z - 3) at 0:", (1/(z - 3)).series(z == 0, 5))
# the culprit: the principal log of e^{it} - 3 jumps at t = 0 (the point -2 lies on the cut)
print("Log(e^{it} - 3) just after t = 0 and just before t = 2 pi:", CC(exp(I*0.001) - 3).log(), CC(exp(I*(2*pi - 0.001)) - 3).log())
