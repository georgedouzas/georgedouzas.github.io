# Exercise exr-wick-gaussian: the Fresnel integral of e^{i t^2} over the real line, by Sage and by the rotation
# t = e^{i pi/4} u to a Gaussian.
t = var("t", domain="real")
val = integrate(exp(I*t^2), t, -oo, oo)
print("integrate:", val, "=", val.n())
rot = exp(I*pi/4)*sqrt(pi)
print("rotation e^{i pi/4} sqrt(pi) =", rot.simplify_full(), "=", rot.n())
print("agree:", abs(val.n() - rot.n()) < 1e-12)
print("real and imaginary parts (the Fresnel integrals of cos t^2 and sin t^2):", integrate(cos(t^2), t, -oo, oo), integrate(sin(t^2), t, -oo, oo))
