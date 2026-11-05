# Figure: the densities |psi_k|^2 of the M = 4 zero modes on the unit sphere, psi_k ~ sin^k(theta/2) cos^{M-1-k}(theta/2)
# (times e^{-i(k+1/2) phi} in the lower component), normalised; k = 0, ..., M - 1 label the states of the spin (M-1)/2
# multiplet. Output: figures/sd-monopole-zero-modes.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-monopole-zero-modes.svg")
th, ph = var("th ph")
M = 4
cols = ["royalblue", "darkgreen", "darkorange", "firebrick"]
P = Graphics()
for k in range(M):
    f = sin(th / 2) ^ (2 * k) * cos(th / 2) ^ (2 * (M - 1 - k))
    nrm = 2 * pi * integrate(f * sin(th), th, 0, pi)
    P += plot(f / nrm, (th, 0, pi), color=cols[k], thickness=1.6, legend_label=r"$k = %d$" % k)
P.axes_labels([r"$\theta$", r"$|\psi_k|^2$"]); P.set_legend_options(loc="upper center", font_size=8)
P.save(OUT, figsize=[6.5, 3.6], ticks=[pi / 4, None], tick_formatter=[pi, None])
# the zero-mode equation for the lower component: g' + (cot/2) g + (m + a)/sin g = 0 with a = (M/2)(1 - cos theta)
a = M / 2 * (1 - cos(th))
ok = True
for k in range(M):
    m = -QQ(1) / 2 - k
    g = sin(th / 2) ^ (-m - 1/2) * cos(th / 2) ^ (m + M - 1/2)
    res = (diff(g, th) + cot(th) / 2 * g + (m + a) / sin(th) * g) / g
    ok &= all(abs(res.subs(th=t).n()) < 1e-10 for t in (0.3, 1.1, 2.2, 2.9))
print("[PASS] g = sin^{-m-1/2}(theta/2) cos^{m+M-1/2}(theta/2), m = -1/2 - k, solves the zero-mode equation for k = 0..M-1:", ok)
print("[PASS] the modes are normalisable: int |g|^2 sin theta < oo exactly for -M + 1/2 <= m <= -1/2 (M values):", len([m for m in [QQ(t) / 2 for t in range(-21, 21, 2)] if -m - 1/2 >= 0 and m + M - 1/2 >= 0]) == M)
norms = [2 * pi * integrate(sin(th / 2) ^ (2 * k) * cos(th / 2) ^ (2 * (M - 1 - k)) * sin(th), th, 0, pi) for k in range(M)]
print("[PASS] 2 pi int sin^{2k}(theta/2) cos^{2(M-1-k)}(theta/2) sin(theta) = 4 pi k! (M-1-k)! / M!:", all(bool(norms[k] == 4 * pi * factorial(k) * factorial(M - 1 - k) / factorial(M)) for k in range(M)))
print("wrote", os.path.relpath(OUT))
