# Figure: the differential of the height function. Left: in the stereographic chart (u, v) the level sets of
# Z = (u^2 + v^2 - 1)/(u^2 + v^2 + 1) are circles; the covector dZ is drawn as the density of level lines, and the
# gradient direction as arrows. Right: on the sphere the level sets are the parallels, and dZ vanishes on the
# rotation field d/dphi (tangent to them) and equals -sin(theta) on d/dtheta. Output: figures/height_levels.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "height_levels.svg")
u, v = var("u v", domain="real")
Z = (u**2 + v**2 - 1)/(u**2 + v**2 + 1)
left = Graphics()
for c in [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]:
    rc = sqrt((1 + c)/(1 - c))
    left += circle((0, 0), rc, color="royalblue" if c else "black", thickness=1.5 if c == 0 else 0.8)
for (x0, y0) in [(0.5, 0.0), (0.0, 1.2), (-1.5, 0.8)]:
    g = vector([diff(Z, u), diff(Z, v)]).subs(u=x0, v=y0)
    left += arrow((x0, y0), (x0 + 0.8*g[0], y0 + 0.8*g[1]), color="firebrick", width=1.5)
left += text(r"level sets $Z=$ const in the $(u,v)$ chart", (0, 3.4), fontsize=10) + text(r"$Z=0$: the equator", (-2.2, -0.35), fontsize=9)
left.axes(False); left.set_aspect_ratio(1)
th = var("th", domain="real")
right = Graphics()
right += circle((0, 0), 1, color="black", thickness=1)
for tk in [pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6]:
    right += line([(-sin(tk), cos(tk)), (sin(tk), cos(tk))], color="royalblue", thickness=1.2)
    right += text(r"$Z=%s$" % latex(cos(tk)), (-sin(tk) - 0.4, cos(tk)), fontsize=8)
P = (sin(pi/3), cos(pi/3))
right += arrow(P, (P[0] + 0.3*cos(pi/3), P[1] - 0.3*sin(pi/3)), color="darkgreen", width=1.5) + text(r"$\partial_\theta$, $dZ(\partial_\theta)=-\sin\theta$", (P[0] + 0.1, P[1] - 0.9), fontsize=8, color="darkgreen")
right += point(P, color="black", size=30) + text(r"$\partial_\varphi\perp$ page, $dZ(\partial_\varphi)=0$", (0, -1.35), fontsize=8, color="royalblue")
right += text("a meridian plane of the sphere", (0, 1.4), fontsize=10)
right.axes(False); right.set_aspect_ratio(1)
g = graphics_array([left, right]); g.save(OUT, figsize=[8, 4])
dZ = vector([diff(Z, u), diff(Z, v)])
print("[PASS] dZ in the (u, v) chart is radial, 4(u, v)/(1 + u^2 + v^2)^2:", bool((dZ - 4*vector([u, v])/(1 + u**2 + v**2)**2).simplify_full() == 0))
print("[PASS] dZ vanishes on the rotation field (-v, u):", bool((dZ * vector([-v, u])).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
