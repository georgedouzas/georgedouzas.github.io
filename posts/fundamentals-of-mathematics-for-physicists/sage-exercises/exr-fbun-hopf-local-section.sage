# Exercise exr-fbun-hopf-local-section: s(theta, phi) = (cos(theta/2), sin(theta/2) e^{-i phi}) is a section of the
# Hopf map over the sphere minus the south pole: pi(s) is the point (theta, phi) of S^2, s is smooth at the north
# pole (independent of phi there) and has no limit at the south pole (it depends on phi there).
th, ph = var("th ph", domain="real")
a = cos(th / 2); b = sin(th / 2) * exp(-I * ph)
def hopf(a, b):
    ab = a * b.conjugate()
    return vector([2 * ab.real(), 2 * ab.imag(), abs(a) ** 2 - abs(b) ** 2])
target = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
print("|a|^2 + |b|^2 =", (abs(a) ** 2 + abs(b) ** 2).simplify_full())
print("pi(s(theta, phi)) = (sin th cos ph, sin th sin ph, cos th):",
      bool(all((e.simplify_full().trig_reduce() == 0) for e in hopf(a, b) - target)))
print("s at the north pole theta = 0:", (a.subs(th=0), b.subs(th=0)), "(independent of phi)")
print("s at the south pole theta = pi:", (a.subs(th=pi), b.subs(th=pi)), "(depends on phi: no continuous extension)")
# a second section, regular at the south pole: s'(theta, phi) = (cos(theta/2) e^{i phi}, sin(theta/2))
a2, b2 = cos(th / 2) * exp(I * ph), sin(th / 2)
print("s' = (cos(th/2) e^{i ph}, sin(th/2)) is also a section:",
      bool(all((e.simplify_full().trig_reduce() == 0) for e in hopf(a2, b2) - target)))
print("s' = e^{i phi} s on the overlap, the transition function:", bool((a2 - exp(I * ph) * a).simplify_full() == 0
      and (b2 - exp(I * ph) * b).simplify_full() == 0))
