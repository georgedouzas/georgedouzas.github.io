# Exercise exr-thd-adiabat: the reversible adiabat of the monatomic ideal gas from (3/2) N dT = -(N T/V) dV, and the
# work done along it, W = (3/2) N (T_1 - T_2).
V, N, T1, V1, V2 = var("V N T_1 V_1 V_2"); assume(V > 0, N > 0, T1 > 0, V1 > 0, V2 > V1)
Tf = function("T")(V)
sol = desolve(diff(Tf, V) == -2 * Tf / (3 * V), Tf, ics=[V1, T1])
print("T(V) =", sol.simplify_log().canonicalize_radical())
print("T V^(2/3) is constant:", bool(diff((sol * V ** (2 / 3)).canonicalize_radical(), V) == 0), "; p V^(5/3) =", (N * sol / V * V ** (5 / 3)).canonicalize_radical())
Tad = T1 * (V1 / V) ** (2 / 3)
print("the solution equals T_1 (V_1/V)^(2/3):", bool((sol - Tad).canonicalize_radical() == 0))
W = integrate(N * Tad / V, V, V1, V2)
T2 = Tad.subs(V=V2)
print("W = int p dV =", W.canonicalize_radical(), "; W - (3/2) N (T_1 - T_2) =", (W - 3 / 2 * N * (T1 - T2)).canonicalize_radical())
