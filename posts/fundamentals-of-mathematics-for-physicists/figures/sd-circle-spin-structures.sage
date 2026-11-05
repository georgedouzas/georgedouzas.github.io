# Figure: the two spin structures of the circle. Spinor fields are periodic (trivial structure) or antiperiodic
# (non-trivial, bounding structure) functions of the angle; the plot shows the real parts of the lowest modes
# e^{i k theta} with k in Z (periodic) and k in Z + 1/2 (antiperiodic) over two turns.
# Output: figures/sd-circle-spin-structures.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-circle-spin-structures.svg")
th = var("th")
P = plot(cos(0 * th), (th, 0, 4 * pi), color="royalblue", thickness=2, legend_label=r"periodic, $k = 0$")
P += plot(cos(th), (th, 0, 4 * pi), color="royalblue", linestyle="--", legend_label=r"periodic, $k = 1$")
P += plot(cos(th / 2), (th, 0, 4 * pi), color="firebrick", thickness=2, legend_label=r"antiperiodic, $k = 1/2$")
P += plot(cos(3 * th / 2), (th, 0, 4 * pi), color="firebrick", linestyle="--", legend_label=r"antiperiodic, $k = 3/2$")
P += line([(2 * pi, -1.25), (2 * pi, 1.25)], color="grey", linestyle=":")
P.axes_labels([r"$\theta$", r"$\mathrm{Re}\,e^{ik\theta}$"]); P.set_legend_options(loc="lower left", font_size=8)
P.save(OUT, figsize=[7, 3.6], ymin=-1.3, ymax=1.3, ticks=[pi, 0.5], tick_formatter=[pi, None])
k = var("k")
print("[PASS] e^{ik theta} with k in Z is periodic and with k in Z + 1/2 antiperiodic:", all(bool(exp(I * kk * (th + 2 * pi)) == exp(I * kk * th)) for kk in (0, 1, 2)) and all(bool((exp(I * kk * (th + 2 * pi)) + exp(I * kk * th)).simplify_full() == 0) for kk in (1/2, 3/2, -1/2)))
print("[PASS] the Dirac operator i d/dtheta has eigenvalues -k: zero mode only for the periodic structure:", all(bool(I * diff(exp(I * kk * th), th) == -kk * exp(I * kk * th)) for kk in (0, 1/2, 1, 3/2)))
print("wrote", os.path.relpath(OUT))
