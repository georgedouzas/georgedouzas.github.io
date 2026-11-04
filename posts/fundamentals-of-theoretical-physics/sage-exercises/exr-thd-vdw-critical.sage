# Exercise exr-thd-vdw-critical: the critical point of the van der Waals gas p = N T/(V - N b) - a N^2/V^2 and the
# reduced equation of state in p_r = p/p_c, V_r = V/V_c, T_r = T/T_c.
p, V, T, N, a, b = var("p V T N a b"); assume(V > 0, T > 0, N > 0, a > 0, b > 0)
P = N * T / (V - N * b) - a * N ** 2 / V ** 2
Tsol = solve(diff(P, V) == 0, T)[0].rhs()
print("(dp/dV)_T = 0 gives T =", Tsol)
eq2 = diff(P, V, 2).subs(T=Tsol).simplify_full()
print("then (d^2p/dV^2)_T = 0 has the roots V =", solve(eq2.numerator() == 0, V))
Vc = 3 * N * b; Tc = Tsol.subs(V=Vc).simplify_full(); pc = P.subs(V=Vc, T=Tc).simplify_full()
print("critical point: V_c =", Vc, " T_c =", Tc, " p_c =", pc, " p_c V_c/(N T_c) =", (pc * Vc / (N * Tc)).simplify_full())
Vr, Tr = var("V_r T_r")
pr = (P.subs(V=Vr * Vc, T=Tr * Tc) / pc).simplify_full()
print("reduced equation of state: p_r =", pr, "; equals 8 T_r/(3 V_r - 1) - 3/V_r^2:", bool((pr - (8 * Tr / (3 * Vr - 1) - 3 / Vr ** 2)).simplify_full() == 0))
