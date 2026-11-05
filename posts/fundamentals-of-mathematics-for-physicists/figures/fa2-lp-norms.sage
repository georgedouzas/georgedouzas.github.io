# Figure: p -> ||a||_p for sequences (decreasing in p) and p -> ||f||_{L^p(0,1)} for functions (increasing in p).
# Output: figures/fa2-lp-norms.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-lp-norms.svg")
p = var("p")
seq1 = lambda q: RR(zeta(q)) ^ (1 / q)                          # a_n = 1/n
seq2 = lambda q: (1 / (2 ^ q - 1)) ^ (1 / q)                     # a_n = 2^{-n}
fun1 = lambda q: (4 / (4 - q)) ^ (1 / q)                         # x^{-1/4}
fun2 = lambda q: (1 / (q + 1)) ^ (1 / q)                         # x
L = plot(seq1, (1.05, 6), color="royalblue", thickness=1.8, legend_label=r"$a_n = 1/n$")
L += plot(seq2, (1, 6), color="darkgreen", thickness=1.8, legend_label=r"$a_n = 2^{-n}$")
L.axes_labels([r"$p$", r"$\|a\|_p$"]); L.set_legend_options(loc="upper right", font_size=9)
R = plot(fun1, (1, 3.8), color="firebrick", thickness=1.8, legend_label=r"$f = x^{-1/4}$")
R += plot(fun2, (1, 6), color="darkorange", thickness=1.8, legend_label=r"$f = x$")
R.axes_labels([r"$p$", r"$\|f\|_{L^p(0,1)}$"]); R.set_legend_options(loc="upper left", font_size=9)
graphics_array([L, R]).save(OUT, figsize=[10, 4])
qs = [1.1, 1.5, 2, 3, 5]
print("[PASS] ||1/n||_p and ||2^-n||_p decrease in p:", all(seq1(a) > seq1(b) and seq2(a) > seq2(b) for a, b in zip(qs, qs[1:])))
print("[PASS] ||x^-1/4||_p and ||x||_p on (0,1) increase in p:", all(fun1(a) < fun1(b) for a, b in zip([1, 1.5, 2, 3], [1.5, 2, 3, 3.5])) and all(fun2(a) < fun2(b) for a, b in zip(qs, qs[1:])))
x = var("x")
print("[PASS] int_0^1 x^{-p/4} dx = 4/(4 - p) at p = 2, 3:", bool(integrate(x ^ (-2 / 4), x, 0, 1) == 2) and bool(integrate(x ^ (-3 / 4), x, 0, 1) == 4))
y = vector([3, 2, 1]); xx = vector([1, 2, 3])
hold = lambda u, v: RR(sum(a ^ 3 for a in u)) ^ (1 / 3) * RR(sum(RR(b) ^ 1.5 for b in v)) ^ (2 / 3)
print("[PASS] Hoelder for x = (1,2,3), y = (3,2,1), p = 3: 10 <= 14.31, equality for y = (1,4,9):", xx.dot_product(y) <= hold(xx, y) and abs(hold(xx, y) - 14.31) < 0.01 and xx.dot_product(vector([1, 4, 9])) == 36 and abs(hold(xx, [1, 4, 9]) - 36) < 1e-9)
print("[PASS] Gaussian norms ||e^{-x^2/2}||_p = (2 pi/p)^{1/(2p)} at p = 1, 2:", all(abs(integrate(exp(-q * x ^ 2 / 2), x, -oo, oo).n() ^ (1 / q) - ((2 * pi / q) ^ (1 / (2 * q))).n()) < 1e-10 for q in [1, 2, 3]))
print("wrote", os.path.relpath(OUT))
