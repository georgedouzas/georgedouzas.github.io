# Figure: errors of the partial sums of the Neumann series for A1 = sigma_1/2 and A2 = [[1/2, 4], [0, 1/2]].
# Output: figures/fa2-neumann-series.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-neumann-series.svg")
A1 = matrix(QQ, [[0, 1/2], [1/2, 0]]); A2 = matrix(QQ, [[1/2, 4], [0, 1/2]])
def errs(A, Ns):
    inv = (1 - A).inverse()
    return [(inv - sum(A ^ k for k in range(N))).change_ring(RDF).norm(2) for N in Ns]
Ns = list(range(1, 41))
e1 = errs(A1, Ns); e2 = errs(A2, Ns)
P = list_plot(list(zip(Ns, e1)), plotjoined=True, color="royalblue", scale="semilogy", legend_label=r"$A_1 = \sigma_1/2$")
P += list_plot(list(zip(Ns, e2)), plotjoined=True, color="firebrick", scale="semilogy", legend_label=r"$A_2$")
P.set_legend_options(loc="upper right", font_size=9); P.axes_labels([r"$N$", "error"])
P.save(OUT, figsize=[6, 4.5])
print("[PASS] the error for A1 is exactly 2^(1-N) at even N:", all(abs(e1[N - 1] - 2.0 ^ (1 - N)) < 1e-12 for N in Ns if N % 2 == 0))
print("[PASS] (1 - A2)^{-1} = [[2, 16], [0, 2]] and the error decays like 8 (N + 1) 2^(1 - N):", (1 - A2).inverse() == matrix(QQ, [[2, 16], [0, 2]]) and all(abs(e2[N - 1] / (8 * (N + 1) * 2.0 ^ (1 - N)) - 1) < 0.01 for N in range(20, 41)))
print("[PASS] ||A2|| > 4, the error for A2 decreases slowly at first and drops below 1 at N = 8:", A2.change_ring(RDF).norm(2) > 4 and min(N for N in Ns if e2[N - 1] < 1) == 8)
print("wrote", os.path.relpath(OUT))
