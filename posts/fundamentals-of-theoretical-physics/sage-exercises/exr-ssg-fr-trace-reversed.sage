# Exercise exr-ssg-fr-trace-reversed: trace-reverse the eleven-dimensional Einstein equation
# R_MN - (1/2) g_MN R = -(1/3)(FF_MN - (1/8) g_MN F^2), FF_MN = F_MPQR F_N^PQR, in D = 11. Taking the trace (g^MN g_MN = 11,
# g^MN FF_MN = F^2) gives R = -F^2/36 and R_MN = -(1/3)(FF_MN - (1/12) g_MN F^2). Then evaluate on the Freund-Rubin ansatz,
# FF_{mu nu} = -54 m^2 g_{mu nu}, FF_mn = 0, F^2 = -216 m^2.
D = 11; FF, F2, g, R = var("FF F2 g R")       # a single component FF_MN with its metric component g_MN, treated generically
trace_eq = R - D * R / 2 == -(F2 - D * F2 / 8) / 3
Rsol = solve(trace_eq, R)[0].rhs()
print("R =", Rsol)
print("[PASS] R = -F^2/36:", bool(Rsol == -F2 / 36))
Ric = -(FF - g * F2 / 8) / 3 + g * Rsol / 2
print("[PASS] R_MN = -(1/3)(FF_MN - g_MN F^2/12):", bool((Ric - (-(FF - g * F2 / 12) / 3)).expand() == 0))
m = var("m")
Rst = Ric.subs(FF=-54 * m^2 * g, F2=-216 * m^2); Rint = Ric.subs(FF=0, F2=-216 * m^2)
print("on the ansatz: R_{mu nu} =", Rst.expand(), " R_mn =", Rint.expand(), " R =", Rsol.subs(F2=-216 * m^2))
print("[PASS] R_{mu nu} = 12 m^2 g_{mu nu}, R_mn = -6 m^2 g_mn and R = 6 m^2, as in the proposition:",
      bool(Rst.expand() == 12 * m^2 * g) and bool(Rint.expand() == -6 * m^2 * g) and bool(Rsol.subs(F2=-216 * m^2) == 6 * m^2))
print("[PASS] consistency: R = 4 * 12 m^2 + 7 * (-6 m^2) = 6 m^2 (traces over the two factors):", 4 * 12 - 7 * 6 == 6)
