# Figure: stereographic projection from the north pole, in a plane through the axis: the point p of the circle (a
# meridian of the sphere) is sent to the point where the line from N through p meets the equatorial plane; the
# chart covers the sphere minus N. Output: figures/stereographic.svg (2D, vector).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stereographic.svg")
NP = (0, 1)
p = circle((0, 0), 1, color="gray") + line([(-2.6, 0), (2.6, 0)], color="black")
p += point(NP, size=50, color="black") + text("N", (0.1, 1.12), fontsize=12)
for th in [pi/4, pi/2, 3*pi/4, pi - 0.35]:
    P = (sin(th), cos(th)); u = sin(th) / (1 - cos(th))        # u = X / (1 - Z)
    p += line([NP, (u, 0)], color="royalblue", linestyle="--") + point(P, size=40, color="firebrick") + point((u, 0), size=40, color="royalblue")
p += text("p", (sin(pi/4) + 0.12, cos(pi/4) + 0.08), fontsize=11, color="firebrick") + text("u", (sin(pi/4)/(1-cos(pi/4)) + 0.05, -0.15), fontsize=11, color="royalblue")
p += text("S", (0.1, -1.15), fontsize=12)
p.set_aspect_ratio(1); p.axes(False)
p.save(OUT, figsize=6)
th = var("th")
u = sin(th) / (1 - cos(th))
print("[PASS] u = X/(1 - Z) = cot(theta/2):", all(abs(N((u - cot(th / 2)).subs(th=v))) < 1e-12 for v in [0.3, 1.1, 2.0, 2.9]))
print("[PASS] u -> infinity at the north pole and u = 0 at the south pole:", bool(limit(u, th=0, dir="+") == +infinity) and bool(u.subs(th=pi) == 0))
print("wrote", os.path.relpath(OUT))
