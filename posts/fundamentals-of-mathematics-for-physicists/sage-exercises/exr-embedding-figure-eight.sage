# Exercise exr-embedding-figure-eight: the figure eight t -> (sin 2t, sin t) on (0, 2 pi) is injective and has rank one
# everywhere, but it is not a homeomorphism onto its image: both ends t -> 0 and t -> 2 pi approach the crossing point
# E(pi) = (0, 0), so it is not an embedding.
t = var("t", domain="real")
E = vector([sin(2*t), sin(t)])
V = diff(E, t)
print("velocity:", V)
print("if cos t = 0 then cos 2t = 2 cos^2 t - 1 =", (2*cos(t)**2 - 1).subs(cos(t) == 0), " != 0, so the velocity never vanishes: rank one everywhere")
# injectivity: sin s = sin t forces s = t or s = pi - t (mod 2 pi); on the second branch
# sin 2s = sin(2 pi - 2t) = -sin 2t, so a coincidence needs sin 2t = 0, that is t in {pi/2, pi, 3 pi/2}
print("on the branch s = pi - t: sin 2s + sin 2t =", (sin(2*(pi - t)) + sin(2*t)).simplify_full(), " so E(s) = E(t) needs sin 2t = 0")
for t0 in [pi/2, pi, 3*pi/2]:
    s0 = pi - t0
    s0 = s0 + 2*pi if bool(s0 <= 0) else s0
    print("  t =", t0, ": partner s = pi - t mod 2 pi =", s0, "; same parameter:", bool(s0 == t0), "; partner inside (0, 2 pi):", bool(0 < s0 < 2*pi))
print("hence E is injective on (0, 2 pi)")
print("limits at the two ends of the interval:", (limit(E[0], t=0, dir="+"), limit(E[1], t=0, dir="+")), (limit(E[0], t=2*pi, dir="-"), limit(E[1], t=2*pi, dir="-")), " both equal E(pi) =", E.subs(t=pi))
print("so the inverse E^{-1} is not continuous at (0, 0): points near the origin on the image have parameters near 0, near pi and near 2 pi")
