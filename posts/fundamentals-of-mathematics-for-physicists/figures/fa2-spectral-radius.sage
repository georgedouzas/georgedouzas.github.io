# Figure: ||a^n||^{1/n} for a non-normal 2 x 2 matrix, the matrix of Izaac and Wang (Problem P7.6) and the Volterra
# operator (||V^n|| = 1/n!). Output: figures/fa2-spectral-radius.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-spectral-radius.svg")
A = matrix(QQ, [[1/2, 4], [0, 1/2]]); B = matrix(QQ, [[0, 9, 0], [1, 0, 0], [0, 2, 2]])
ns = list(range(1, 61))
a = [(A ^ n).change_ring(RDF).norm(2) ^ (1.0 / n) for n in ns]
b = [(B ^ n).change_ring(RDF).norm(2) ^ (1.0 / n) for n in ns]
v = [(1 / RR(factorial(n))) ^ (1.0 / n) for n in ns]
P = list_plot(list(zip(ns, a)), plotjoined=True, color="royalblue", legend_label=r"$A$, $r = 1/2$")
P += list_plot(list(zip(ns, b)), plotjoined=True, color="firebrick", legend_label=r"$B$, $r = 3$")
P += list_plot(list(zip(ns, v)), plotjoined=True, color="darkgreen", legend_label=r"$V$, $r = 0$")
for r, c in [(0.5, "royalblue"), (3, "firebrick"), (0, "darkgreen")]:
    P += line([(1, r), (60, r)], color=c, linestyle="dashed")
P.set_legend_options(loc="upper right", font_size=9); P.axes_labels([r"$n$", r"$\|a^n\|^{1/n}$"])
P.save(OUT, figsize=[6, 4.5], ymax=5)
print("[PASS] eigenvalues: A has 1/2 (double), B has 3, -3, 2:", A.eigenvalues() == [1/2, 1/2] and sorted(B.eigenvalues()) == [-3, 2, 3])
print("[PASS] ||A^n||^{1/n} -> 1/2 and ||B^n||^{1/n} -> 3 decreasingly towards the limit (n = 60 within 12%):", abs(a[-1] - 0.5) < 0.06 and abs(b[-1] - 3) < 0.3 and all(x >= 0.5 for x in a) and all(x >= 3 - 1e-9 for x in b))
x, t = var("x t")
print("[PASS] V^n 1 = x^n/n! (n = 1..4), so ||V^n||_{C -> C} = 1/n!:", all(bool(integrate(t ^ (n - 1) / factorial(n - 1), t, 0, x) == x ^ n / factorial(n)) for n in range(1, 5)))
print("[PASS] r(a) <= ||a^n||^{1/n} for every n (A and B):", all(x >= 0.5 - 1e-12 for x in a) and all(x >= 3 - 1e-9 for x in b))
print("wrote", os.path.relpath(OUT))
