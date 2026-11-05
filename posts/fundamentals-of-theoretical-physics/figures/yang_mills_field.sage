# Figure: a non-abelian field strength without derivatives. For the constant SU(2) potential A_x = c1 T_1,
# A_y = c2 T_2 the field strength F_xy = [A_x, A_y] = c1 c2 T_3 is non-zero: the panel shows |F^3_xy| = |c1 c2|
# over the (c1, c2) plane, with the abelian case (F = 0 for every constant potential) as the flat axes. Computed
# from the su(2) structure constants with Sage. Output: figures/yang_mills_field.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yang_mills_field.svg")
c1, c2 = var("c1 c2", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
Ax, Ay = c1 * T[0], c2 * T[1]
F = Ax * Ay - Ay * Ax
F3 = (-2 * (T[2] * F).trace()).simplify_full()               # component along T_3 with tr(T_a T_b) = -delta_ab/2
p = density_plot(abs(F3), (c1, -2, 2), (c2, -2, 2), cmap="Reds", plot_points=80)
p += contour_plot(F3, (c1, -2, 2), (c2, -2, 2), contours=[-2, -1, -0.5, 0.5, 1, 2], fill=False, cmap="Blues", linewidths=1)
p += line([(-2, 0), (2, 0)], color="black") + line([(0, -2), (0, 2)], color="black")
p += text(r"$F^3_{xy}=c_1c_2$ for $A_x=c_1T_1$, $A_y=c_2T_2$: zero only on the axes", (0, 2.3), fontsize=9)
p.axes_labels([r"$c_1$", r"$c_2$"]); p.set_aspect_ratio(1)
p.save(OUT, figsize=5)
print("[PASS] the field strength of the constant potential is [A_x, A_y] = c1 c2 T_3:", bool(F == c1 * c2 * T[2]))
print("[PASS] for an abelian potential (both components along T_3) the commutator vanishes:", bool((c1 * T[2]) * (c2 * T[2]) - (c2 * T[2]) * (c1 * T[2]) == 0))
print("wrote", os.path.relpath(OUT))
