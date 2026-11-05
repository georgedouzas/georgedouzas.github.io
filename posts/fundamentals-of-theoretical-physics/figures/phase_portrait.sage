# Figure: phase portraits. Left: the harmonic oscillator H = (p^2 + q^2)/2, whose orbits are the circles H = const,
# traversed clockwise at unit angular speed. Right: the pendulum H = p^2/2 - cos q, with closed orbits (libration)
# inside the separatrix H = 1 (red) and open orbits (rotation) outside. Drawn with Sage's contour and vector-field
# plots from Hamilton's equations. Output: figures/phase_portrait.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_portrait.svg")
q, p = var("q p", domain="real")
H1 = (p**2 + q**2) / 2
left = contour_plot(H1, (q, -2.2, 2.2), (p, -2.2, 2.2), contours=[0.125, 0.5, 1.125, 2.0], fill=False, cmap="Blues", linewidths=1.5)
left += plot_vector_field((diff(H1, p), -diff(H1, q)), (q, -2.2, 2.2), (p, -2.2, 2.2), color="grey", plot_points=9)
left += text(r"oscillator: $H=\frac{1}{2}(p^2+q^2)$, orbits are circles", (0, 2.5), fontsize=9)
left.axes_labels([r"$q$", r"$p$"]); left.set_aspect_ratio(1)
H2 = p**2 / 2 - cos(q)
right = contour_plot(H2, (q, -2 * pi, 2 * pi), (p, -3, 3), contours=[-0.8, -0.3, 0.3, 1.5, 2.5], fill=False, cmap="Blues", linewidths=1.2)
right += contour_plot(H2, (q, -2 * pi, 2 * pi), (p, -3, 3), contours=[1.0], fill=False, cmap=["firebrick"], linewidths=2)
right += plot_vector_field((diff(H2, p), -diff(H2, q)), (q, -2 * pi, 2 * pi), (p, -3, 3), color="grey", plot_points=11)
right += text(r"pendulum: $H=\frac{1}{2}p^2-\cos q$, separatrix $H=1$ (red)", (0, 3.4), fontsize=9)
right.axes_labels([r"$q$", r"$p$"])
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 4.2]); fig.savefig(OUT, bbox_inches="tight")
t = var("t", domain="real")
qs, ps = cos(t), -sin(t)
print("[PASS] (q, p) = (cos t, -sin t) solves Hamilton's equations of the oscillator and conserves H:", bool(diff(qs, t) == diff(H1, p).subs(q=qs, p=ps)) and bool(diff(ps, t) == -diff(H1, q).subs(q=qs, p=ps)) and bool(H1.subs(q=qs, p=ps).simplify_full() == 1/2))
print("[PASS] H is conserved along every solution of Hamilton's equations (dH/dt = H_q q' + H_p p' = 0):", bool((diff(H2, q) * diff(H2, p) + diff(H2, p) * (-diff(H2, q))).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
