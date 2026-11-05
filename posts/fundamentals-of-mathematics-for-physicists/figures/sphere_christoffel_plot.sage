# Figure: the two non-zero Christoffel symbols of the round sphere as functions of the colatitude theta,
# Gamma^theta_phiphi = -sin(theta) cos(theta) and Gamma^phi_thetaphi = cos(theta)/sin(theta), computed with
# SageManifolds' christoffel_symbols. The second diverges at the poles, where the chart degenerates.
# Output: figures/sphere_christoffel_plot.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sphere_christoffel_plot.svg")
S2 = Manifold(2, "S^2", structure="Riemannian"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
g = A.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th)**2
Gam = g.christoffel_symbols()
G_thpp = Gam[0, 1, 1].expr(); G_phtp = Gam[1, 0, 1].expr()
p = plot(G_thpp, (th, 0, pi), color="firebrick", thickness=2, legend_label=r"$\Gamma^\theta_{\varphi\varphi}=-\sin\theta\cos\theta$")
p += plot(G_phtp, (th, 0.15, pi - 0.15), color="royalblue", thickness=2, legend_label=r"$\Gamma^\varphi_{\theta\varphi}=\cot\theta$", ymin=-4, ymax=4)
p += line([(pi / 2, -4), (pi / 2, 4)], color="grey", linestyle="dashed") + text("equator", (pi / 2 + 0.25, 3.6), fontsize=9, color="grey")
p.set_legend_options(loc="upper right", font_size=9)
p.axes_labels([r"$\theta$", ""])
p.save(OUT, figsize=[6, 3.6])
print("[PASS] Gamma^theta_phiphi = -sin cos and Gamma^phi_thetaphi = cos/sin from SageManifolds:", bool((G_thpp + sin(th) * cos(th)).simplify_full() == 0 and (G_phtp - cos(th) / sin(th)).simplify_full() == 0))
nonzero = [(k, i, j) for k in range(2) for i in range(2) for j in range(i, 2) if Gam[k, i, j].expr() != 0]
print("[PASS] exactly two independent symbols are non-zero:", nonzero == [(0, 1, 1), (1, 0, 1)])
print("wrote", os.path.relpath(OUT))
