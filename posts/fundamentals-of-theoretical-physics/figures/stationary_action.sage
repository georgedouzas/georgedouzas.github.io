# Figure: the principle of stationary action for the harmonic oscillator. Left: the classical path x(t) = cos t on
# [0, pi/2] from x = 1 to x = 0 (black) and varied paths x + eps sin(2t) (blue, eps = -0.6 ... 0.6) with the same
# endpoints. Right: the action S(eps) = int (x'^2 - x^2)/2 dt of the varied paths, a parabola with its minimum at
# eps = 0, the classical path. Computed symbolically with Sage. Output: figures/stationary_action.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stationary_action.svg")
t, eps = var("t eps", domain="real")
xcl = cos(t); eta = sin(2 * t); T = pi / 2
x = xcl + eps * eta
L = (diff(x, t)**2 - x**2) / 2
S = integrate(L, t, 0, T).simplify_full()
left = plot(xcl, (t, 0, T), color="black", thickness=2.5, legend_label=r"classical path $\cos t$")
for e in [-0.6, -0.3, 0.3, 0.6]:
    left += plot(x.subs(eps=e), (t, 0, T), color="royalblue", alpha=0.6, thickness=1)
left += text(r"varied paths $\cos t+\varepsilon\sin 2t$ (blue)", (0.8, 1.15), fontsize=9, color="royalblue")
left.axes_labels([r"$t$", r"$x$"]); left.set_legend_options(loc="lower left", font_size=8)
right = plot(S, (eps, -0.8, 0.8), color="firebrick", thickness=2)
right += point((0, S.subs(eps=0)), color="black", size=40) + text(r"$S(\varepsilon)$ is stationary at $\varepsilon=0$", (0, S.subs(eps=0.8) * 0.9), fontsize=9)
right.axes_labels([r"$\varepsilon$", r"$S$"])
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9, 3.8]); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] S(eps) is a quadratic polynomial in eps with no linear term, dS/deps = 0 at eps = 0:", bool(diff(S, eps).subs(eps=0) == 0) and bool(S.degree(eps) == 2))
print("[PASS] the classical path solves x'' + x = 0:", bool(diff(xcl, t, 2) + xcl == 0))
print("wrote", os.path.relpath(OUT))
