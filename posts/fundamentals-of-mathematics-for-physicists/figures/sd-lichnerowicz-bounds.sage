# Figure: lower bounds for the square of the Dirac eigenvalues on the unit sphere S^n: the Lichnerowicz bound R/4 =
# n(n-1)/4, the sharper bound n R / (4 (n - 1)) = n^2/4 (from |D psi|^2 <= n |nabla psi|^2), and the actual smallest
# eigenvalue squared (n/2)^2. Output: figures/sd-lichnerowicz-bounds.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-lichnerowicz-bounds.svg")
ns = list(range(2, 11))
P = list_plot([(n, n * (n - 1) / 4) for n in ns], color="royalblue", size=40, legend_label=r"$R/4 = n(n-1)/4$")
P += list_plot([(n, n ^ 2 / 4) for n in ns], color="firebrick", marker="s", size=40, legend_label=r"$\lambda_{\min}^2 = (n/2)^2 = nR/(4(n-1))$")
P += list_plot([(n, n ^ 2 / 4) for n in ns], plotjoined=True, color="firebrick", linestyle="--")
P += list_plot([(n, n * (n - 1) / 4) for n in ns], plotjoined=True, color="royalblue", linestyle=":")
P.axes_labels([r"$n$", ""]); P.set_legend_options(loc="upper left", font_size=8)
P.save(OUT, figsize=[6, 3.8], ymin=0)
n = var("n")
print("[PASS] n R / (4 (n - 1)) with R = n (n - 1) equals (n/2)^2:", bool((n * n * (n - 1) / (4 * (n - 1))).simplify_full() == n ^ 2 / 4))
print("[PASS] the Lichnerowicz bound R/4 lies below (n/2)^2 by n/4:", bool((n ^ 2 / 4 - n * (n - 1) / 4).simplify_full() == n / 4))
print("[PASS] the smallest eigenvalue l + n/2 at l = 0 is n/2:", all((0 + QQ(k) / 2) ^ 2 == QQ(k) ^ 2 / 4 for k in ns))
print("wrote", os.path.relpath(OUT))
