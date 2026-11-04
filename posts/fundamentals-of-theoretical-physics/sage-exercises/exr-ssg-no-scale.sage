# Exercise exr-ssg-no-scale: the supergravity potential V = e^K (K^{T Tbar} |D_T W|^2 - 3|W|^2) (units M_P = 1) for the
# no-scale Kahler potential K = -3 log(T + Tbar) and a constant superpotential W = W0 vanishes identically, although
# D_T W != 0 (supersymmetry is broken) and m_{3/2} = e^{K/2}|W0| != 0. T and Tbar are treated as independent variables.
T, Tb, W0, W0b, n = var("T Tb W0 W0b n")
def potential(K, W, Wb):
    KT = diff(K, T); KTb = diff(K, Tb); KTTb = diff(K, T, Tb)
    DW = diff(W, T) + KT * W; DWb = diff(Wb, Tb) + KTb * Wb
    return (exp(K) * (DW * DWb / KTTb - 3 * W * Wb)).simplify_full(), DW
V, DW = potential(-3 * log(T + Tb), W0, W0b)
print("K = -3 log(T + Tbar): K_{T Tbar} =", diff(-3 * log(T + Tb), T, Tb), "  D_T W =", DW, "  V =", V)
print("[PASS] the no-scale potential vanishes for every T:", bool(V == 0))
print("[PASS] D_T W = -3 W0/(T + Tbar) is non-zero, so supersymmetry is broken with zero vacuum energy:", bool((DW + 3 * W0 / (T + Tb)).simplify_full() == 0))
print("[PASS] the gravitino mass e^{K/2}|W0| = |W0|/(T + Tbar)^{3/2} depends on the flat direction T:", bool((exp(-3 * log(T + Tb) / 2) - (T + Tb)^(-3/2)).simplify_full() == 0))
# Nath, Problem 12.1, for other powers: K = -n log(T + Tbar)
Vn, _ = potential(-n * log(T + Tb), W0, W0b)
print("K = -n log(T + Tbar): V =", Vn.factor())
print("[PASS] among the potentials K = -n log(T + Tbar), W = W0, only n = 3 gives V = 0:", bool(Vn.subs(n=3) == 0) and bool(Vn.subs(n=1) != 0))
