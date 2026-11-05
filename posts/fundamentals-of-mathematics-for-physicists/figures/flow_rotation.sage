# Figure: flows of two vector fields on the plane. Left: the rotation field X = -y d/dx + x d/dy; its flow Phi_t
# moves every point along a circle by the angle t, and a cloud of points (grey) is shown after times t = 0, 0.6,
# 1.2 (blue, darker with time). Right: the radial field r d/dr = x d/dx + y d/dy; its flow is the dilation by
# e^t, which stretches the cloud. The flows are computed by solving dx/dt = X(x) symbolically with Sage's desolve.
# Output: figures/flow_rotation.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow_rotation.svg")
t = var("t", domain="real")
x = function("x")(t); y = function("y")(t)
rot = desolve_system([diff(x, t) == -y, diff(y, t) == x], [x, y], ics=[0, 1, 0], ivar=t)
dil = desolve_system([diff(x, t) == x, diff(y, t) == y], [x, y], ics=[0, 1, 0], ivar=t)
x0, y0 = var("x0 y0", domain="real")
def flow_rot(px, py, tt):          # Phi_t(p) for the rotation field: rotation by the angle t
    return (px * cos(tt) - py * sin(tt), px * sin(tt) + py * cos(tt))
def flow_dil(px, py, tt):          # Phi_t(p) for the radial field: dilation by e^t
    return (px * exp(tt), py * exp(tt))
cloud = [(1.0 + 0.25 * cos(k * 2 * pi / 8), 0.3 + 0.25 * sin(k * 2 * pi / 8)) for k in range(8)] + [(1.0, 0.3)]
def panel(flow, field, title, times, rng):
    p = plot_vector_field(field, (x0, -rng, rng), (y0, -rng, rng), color="lightgrey", plot_points=12)
    cols = [(0.6, 0.6, 0.6), (0.55, 0.65, 0.95), (0.25, 0.41, 0.88), (0.05, 0.15, 0.6)]
    for i, tt in enumerate(times):
        pts = [flow(px, py, tt) for (px, py) in cloud]
        p += points(pts, color=cols[i], size=18)
    for (px, py) in cloud[:1]:
        p += parametric_plot(flow(px, py, t), (t, 0, times[-1]), color="firebrick", thickness=1.5)
    p += text(title, (0, rng + 0.55), fontsize=10)
    p.axes(False); p.set_aspect_ratio(1)
    return p
left = panel(flow_rot, (-y0, x0), r"rotation field $-y\,\partial_x + x\,\partial_y$: $\Phi_t$ = rotation by $t$", [0, 0.6, 1.2, 1.8], 1.7)
right = panel(flow_dil, (x0, y0), r"radial field $x\,\partial_x + y\,\partial_y$: $\Phi_t$ = dilation by $e^t$", [0, 0.3, 0.6, 0.9], 2.8)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9, 4.3], axes=False); fig.savefig(OUT, bbox_inches="tight")
s = var("s", domain="real")
print("[PASS] the solution of the rotation flow from (1, 0) is (cos t, sin t):", bool(rot[0].rhs() == cos(t) and rot[1].rhs() == sin(t)))
print("[PASS] the flows form one-parameter groups, Phi_s(Phi_t(p)) = Phi_{s+t}(p):", bool(all((a - b).simplify_full() == 0 for a, b in zip(flow_rot(*flow_rot(x0, y0, t), s), flow_rot(x0, y0, s + t)))) and bool(all((a - b).simplify_full() == 0 for a, b in zip(flow_dil(*flow_dil(x0, y0, t), s), flow_dil(x0, y0, s + t)))))
print("wrote", os.path.relpath(OUT))
