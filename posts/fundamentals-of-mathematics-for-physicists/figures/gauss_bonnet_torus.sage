# Figure: the scalar curvature R = 2K of the embedded torus as a function of the tube angle a, with the area element
# sqrt(det g) = r (R + r cos a): the integral of K dA over the torus vanishes (Gauss-Bonnet with Euler
# characteristic 0), computed with SageManifolds' ricci_scalar. Output: figures/gauss_bonnet_torus.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauss_bonnet_torus.svg")
R, r = 2, QQ(4) / 5
T2 = Manifold(2, "T^2", structure="Riemannian"); ab = T2.chart("a:(0,2*pi) b:(0,2*pi)"); a, b = ab[:]
g = T2.metric("g"); g[0, 0] = r**2; g[1, 1] = (R + r * cos(a))**2
Rs = g.ricci_scalar().expr().simplify_full()
dA = sqrt(g.determinant().expr()).simplify_full()           # r (R + r cos a), positive since R > r
p = plot(Rs, (a, 0, 2 * pi), color="firebrick", thickness=2, legend_label=r"$R(a)=2\cos a/(r(R+r\cos a))$")
p += plot(Rs * dA / 2, (a, 0, 2 * pi), color="royalblue", thickness=2, legend_label=r"$K\sqrt{\det g}=\cos a$")
p += line([(0, 0), (2 * pi, 0)], color="grey")
p += text(r"$\int K\,dA = 2\pi\int_0^{2\pi}\cos a\,da = 0 = 2\pi\chi(T^2)$", (pi, 1.45), fontsize=10)
p.set_legend_options(loc="lower right", font_size=9); p.axes_labels([r"$a$", ""])
p.save(OUT, figsize=[6, 3.6])
print("[PASS] scalar curvature of the torus R = 2 cos a / (r (R + r cos a)):", bool((Rs - 2 * cos(a) / (r * (R + r * cos(a)))).simplify_full() == 0))
print("[PASS] Gauss-Bonnet: integral of K dA over the torus is 0:", bool(integrate(integrate(Rs * dA / 2, a, 0, 2 * pi), b, 0, 2 * pi) == 0))
print("wrote", os.path.relpath(OUT))
