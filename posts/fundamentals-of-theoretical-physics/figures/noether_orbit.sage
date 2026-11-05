# Figure: conservation of angular momentum for a central force (Kepler problem). The orbit of a particle in the
# potential V = -1/r, integrated numerically from Hamilton's equations with Sage's desolve_system_rk4, with the
# radius vector drawn at equal time intervals: it sweeps equal areas (Kepler's second law), the content of
# L = x p_y - y p_x = const, the Noether charge of rotations. Output: figures/noether_orbit.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "noether_orbit.svg")
x, y, px, py, t = var("x y px py t")
r3 = (x**2 + y**2)**(3/2)
sol = desolve_system_rk4([px, py, -x / r3, -y / r3], [x, y, px, py], ics=[0, 1.0, 0.0, 0.0, 1.2], ivar=t, end_points=9.0, step=0.01)
pts = [(s[1], s[2]) for s in sol]
p = line(pts, color="royalblue", thickness=1.5) + point((0, 0), color="black", size=50)
Ls = [s[1] * s[4] - s[2] * s[3] for s in sol]
for k in range(0, len(sol), 60):
    p += line([(0, 0), (sol[k][1], sol[k][2])], color="lightsalmon", thickness=1)
for k in range(0, len(sol) - 60, 60):
    seg = [(0, 0)] + [(s[1], s[2]) for s in sol[k:k + 61]]
    p += polygon(seg, color="lightsalmon", alpha=0.25)
p += text(r"$V=-1/r$: equal areas in equal times, $L=xp_y-yp_x$ constant", (-0.8, -0.85), fontsize=9)
p.axes_labels([r"$x$", r"$y$"]); p.set_aspect_ratio(1)
p.save(OUT, figsize=5.5)
print("[PASS] the numerical angular momentum stays constant to 1e-6 along the orbit:", max(Ls) - min(Ls) < 1e-6)
Lsym = x * py - y * px
dL = diff(Lsym, x) * px + diff(Lsym, y) * py + diff(Lsym, px) * (-x / r3) + diff(Lsym, py) * (-y / r3)
print("[PASS] dL/dt vanishes identically for any central force:", bool(dL.simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
