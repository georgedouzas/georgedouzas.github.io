# Figure: decay of the Fourier coefficients of the square wave, the triangle wave |x| and e^{cos x} on the circle.
# Output: figures/fa2-sobolev-torus.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-sobolev-torus.svg")
ks = [k for k in range(1, 60, 2)]
sq = [2 / (pi.n() * k) for k in ks]
tri = [2 / (pi.n() * k ^ 2) for k in ks]
ec = [RDF(bessel_I(k, 1)) for k in range(1, 16)]
P = list_plot(list(zip(ks, sq)), color="royalblue", marker="o", scale="loglog", legend_label=r"square wave, $\propto k^{-1}$")
P += list_plot(list(zip(ks, tri)), color="darkgreen", marker="s", scale="loglog", legend_label=r"$|x|$, $\propto k^{-2}$")
P += list_plot(list(zip(range(1, 16), ec)), color="firebrick", marker="^", scale="loglog", legend_label=r"$e^{\cos x}$")
P.set_legend_options(loc="lower left", font_size=8); P.axes_labels([r"$k$", r"$|c_k|$"])
P.save(OUT, figsize=[6, 4.5], ymin=1e-16)
x = var("x")
c = lambda f, k: numerical_integral(f * cos(k * x), -pi, pi)[0] / (2 * pi.n())
s = lambda f, k: numerical_integral(f * sin(k * x), -pi, pi)[0] / (2 * pi.n())
print("[PASS] square wave: |c_k| = 2/(pi k) for odd k, 0 for even k (k <= 6):", all(abs(abs(s(sign(x), k)) - (2 / (pi.n() * k) if k % 2 else 0)) < 1e-8 for k in range(1, 7)))
print("[PASS] triangle wave: c_k = ((-1)^k - 1)/(pi k^2), c_0 = pi/2:", all(abs(c(abs(x), k) - ((-1) ^ k - 1) / (pi.n() * k ^ 2)) < 1e-8 for k in range(1, 7)) and abs(c(abs(x), 0) - pi.n() / 2) < 1e-10)
print("[PASS] e^{cos x}: c_k = I_k(1):", all(abs(c(exp(cos(x)), k) - RDF(bessel_I(k, 1))) < 1e-10 for k in range(0, 6)))
print("[PASS] ||sin x + cos(2x)/2||_{H^1}^2 = 13 pi/4:", bool(integrate((sin(x) + cos(2 * x) / 2) ^ 2 + diff(sin(x) + cos(2 * x) / 2, x) ^ 2, x, 0, 2 * pi) == 13 * pi / 4))
print("wrote", os.path.relpath(OUT))
