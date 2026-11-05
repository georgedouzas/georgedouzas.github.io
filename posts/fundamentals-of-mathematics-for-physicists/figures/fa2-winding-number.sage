# Figure: the curves phi(e^{i theta}) for phi = z, z^2, z - 1/2, 1/z, 2 + z and their winding numbers about 0.
# Output: figures/fa2-winding-number.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-winding-number.svg")
th = var("th")
syms = [("z", lambda w: w, 1), ("z^2", lambda w: w ^ 2, 2), ("z - 1/2", lambda w: w - 1 / 2, 1), ("z^{-1}", lambda w: 1 / w, -1), ("2 + z", lambda w: 2 + w, 0)]
plots = []
def wind(f, M=4000):
    tot = 0.0
    prev = CDF(f(CDF(1, 0)))
    for k in range(1, M + 1):
        cur = CDF(f(CDF(exp(2 * pi.n() * I * k / M))))
        tot += float((cur / prev).argument())
        prev = cur
    return round(tot / (2 * pi.n()))
ws = []
for name, f, w in syms:
    pts = [CDF(f(CDF(exp(2 * pi.n() * I * k / 400)))) for k in range(401)]
    G = line([(p.real(), p.imag()) for p in pts], color="royalblue", thickness=1.6)
    G += arrow((pts[50].real(), pts[50].imag()), (pts[56].real(), pts[56].imag()), color="royalblue", width=1.5)
    G += point((0, 0), color="firebrick", size=30)
    G += text(r"$\varphi = %s$, wind $= %d$" % (name, w), (0, 3.4), fontsize=9, color="black")
    G.set_aspect_ratio(1)
    plots.append(G)
    ws.append(wind(f))
graphics_array(plots, 1, 5).save(OUT, figsize=[14, 3.4], xmin=-3.2, xmax=3.2, ymin=-3.2, ymax=3.8)
print("[PASS] winding numbers by the argument increments are 1, 2, 1, -1, 0:", ws == [1, 2, 1, -1, 0])
print("[PASS] wind of z^2 (z - 1/2)/(z - 3) is 3:", wind(lambda w: w ^ 2 * (w - 1 / 2) / (w - 3)) == 3)
S = matrix(QQ, 6, 6, lambda i, j: 1 if i == j + 1 else 0)
print("[PASS] the truncated T_z = S has rank 5 and kernel of S^T spanned by e_0 (cokernel of the shift):", S.rank() == 5 and S.transpose().right_kernel().basis() == [vector(QQ, [1, 0, 0, 0, 0, 0])])
print("wrote", os.path.relpath(OUT))
