# Figure: the generating function (x/2)/sinh(x/2) of the A-hat genus and its Taylor polynomials 1 - x^2/24 and
# 1 - x^2/24 + 7x^4/5760. Output: figures/sd-ahat-series.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-ahat-series.svg")
x = var("x")
f = (x / 2) / sinh(x / 2)
P = plot(f, (x, -8, 8), color="black", thickness=2, legend_label=r"$\frac{x/2}{\sinh(x/2)}$")
P += plot(1 - x ^ 2 / 24, (x, -8, 8), color="royalblue", legend_label=r"$1 - \frac{x^2}{24}$")
P += plot(1 - x ^ 2 / 24 + 7 * x ^ 4 / 5760, (x, -8, 8), color="firebrick", linestyle="--", legend_label=r"$1 - \frac{x^2}{24} + \frac{7x^4}{5760}$")
P.set_legend_options(loc="upper right", font_size=8); P.axes_labels([r"$x$", ""])
P.save(OUT, figsize=[6.5, 3.8], ymin=-0.3, ymax=1.6)
T = f.taylor(x, 0, 6)
print("[PASS] (x/2)/sinh(x/2) = 1 - x^2/24 + 7x^4/5760 - 31x^6/967680 + ...:", bool(T == 1 - x ^ 2 / 24 + 7 * x ^ 4 / 5760 - 31 * x ^ 6 / 967680))
x1, x2 = var("x1 x2")
prod2 = (f.subs(x=x1) * f.subs(x=x2)).taylor((x1, 0), (x2, 0), 4)
p1, p2 = x1 ^ 2 + x2 ^ 2, x1 ^ 2 * x2 ^ 2
deg4 = sum(c * x1 ^ i * x2 ^ j for (c, i, j) in [(prod2.coefficient(x1, i).coefficient(x2, j), i, j) for i in range(5) for j in range(5)] if i + j == 4)
print("[PASS] the degree-4 part of prod_j (x_j/2)/sinh(x_j/2) is (7 p1^2 - 4 p2)/5760:", bool((deg4 - (7 * p1 ^ 2 - 4 * p2) / 5760).expand() == 0))
print("[PASS] the degree-2 part is -p1/24:", bool((sum(prod2.coefficient(x1, i).coefficient(x2, j) * x1 ^ i * x2 ^ j for i in range(3) for j in range(3) if i + j == 2) + p1 / 24).expand() == 0))
print("wrote", os.path.relpath(OUT))
