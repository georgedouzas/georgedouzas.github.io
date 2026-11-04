# Exercise exr-thd-vdw-energy: the energy equation (dU/dV)_T = T (dp/dT)_V - p for the van der Waals gas, its internal
# energy U = (3/2) N T - a N^2/V, and the temperature drop in a free (Joule) expansion from V_1 to V_2.
V, T, N, a, b, V1, V2 = var("V T N a b V_1 V_2"); assume(V > 0, T > 0, N > 0, a > 0, b > 0, V1 > 0, V2 > V1)
p = N * T / (V - N * b) - a * N ** 2 / V ** 2
dUdV = (T * diff(p, T) - p).simplify_full()
print("(dU/dV)_T =", dUdV)
U = 3 / 2 * N * T - a * N ** 2 / V
print("U = (3/2) N T - a N^2/V satisfies it:", bool(diff(U, V) == dUdV))
Tdrop = solve(U.subs(V=V1) == U.subs(V=V2, T=var("T2")), var("T2"))[0].rhs()
print("free expansion at constant U: T_2 =", Tdrop, "; T_2 - T =", (Tdrop - T).simplify_full())
