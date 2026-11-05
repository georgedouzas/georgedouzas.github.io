# Figure: the index (Q + 1)(Q + 2)/2 of the spin-c Dirac operator of CP^2 twisted by O(Q) (the holomorphic Euler
# characteristic of O(Q)), for Q = -7, ..., 5, compared with the number of zero modes found from SU(3) Casimirs.
# Output: figures/sd-cp2-index.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-cp2-index.svg")
Qs = list(range(-7, 6))
ind = {Q: (Q + 1) * (Q + 2) / 2 for Q in Qs}
def zero_modes(Q):
    # Dolan (2003), eq. (55): |Y| = 2Q for Q > 0, |Y| = |2Q + 6| for Q <= -3, Y = 0 for Q = 0
    if Q in (-1, -2): return 0
    Y = 2 * Q if Q >= 0 else 2 * Q + 6
    return (abs(Y) + 2) * (abs(Y) + 4) / 8
P = list_plot([(Q, ind[Q]) for Q in Qs], color="royalblue", size=40, legend_label=r"index $(Q+1)(Q+2)/2$")
P += list_plot([(Q, zero_modes(Q)) for Q in Qs], color="firebrick", marker="x", size=60, legend_label=r"number of zero modes")
P += line([(-7.5, 0), (5.5, 0)], color="grey")
P.axes_labels([r"$Q$", ""]); P.set_legend_options(loc="upper center", font_size=8)
P.save(OUT, figsize=[6.5, 3.8], ticks=[1, 5])
x = var("x")
# Riemann-Roch on CP^2: the degree-4 part of ch(O(Q)) td(CP^2) = e^{Qx} (x/(1 - e^{-x}))^3, with int_{CP^2} x^2 = 1
todd = lambda Q: (exp(Q * x) * (x / (1 - exp(-x))) ** 3).taylor(x, 0, 2).coefficient(x, 2)
print("[PASS] Riemann-Roch: int_{CP^2} ch(O(Q)) td(CP^2) = (Q+1)(Q+2)/2 for Q = -7..5:", all(bool(todd(Q) == ind[Q]) for Q in Qs))
print("[PASS] the zero-mode counts from Casimirs equal |index| for every Q:", all(zero_modes(Q) == abs(ind[Q]) for Q in Qs))
print("[PASS] the canonical spin-c structure Q = 0 has exactly one zero mode (the constant section):", ind[0] == 1)
print("wrote", os.path.relpath(OUT))
