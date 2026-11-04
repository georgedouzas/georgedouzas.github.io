# Exercise exr-tsm-photon-carnot: a Carnot cycle run with blackbody radiation (u = sigma' T^4 with sigma' = pi^2/15,
# p = u/3, S = (4/3) sigma' T^3 V). Isotherms are isobars, adiabats have V T^3 constant; the efficiency is 1 - T_c/T_h.
Th, Tc, V1, V2 = var("T_h T_c V_1 V_2"); assume(Th > Tc, Tc > 0, V1 > 0, V2 > V1)
s = pi ** 2 / 15
p = lambda T: s * T ** 4 / 3
U = lambda T, V: s * T ** 4 * V
S = lambda T, V: 4 / 3 * s * T ** 3 * V
# A (V1, Th) -> B (V2, Th) isothermal; B -> C adiabatic to Tc; C -> D isothermal; D -> A adiabatic
V3 = V2 * (Th / Tc) ** 3; V4 = V1 * (Th / Tc) ** 3
print("adiabats: S(B) = S(C):", bool(S(Th, V2) == S(Tc, V3)), "; S(D) = S(A):", bool(S(Tc, V4) == S(Th, V1)))
Qh = U(Th, V2) - U(Th, V1) + p(Th) * (V2 - V1)       # first law on the isotherm (u depends on T only)
Qc = U(Tc, V3) - U(Tc, V4) + p(Tc) * (V3 - V4)
print("Q_h =", Qh.factor(), "= T_h Delta S:", bool((Qh - Th * (S(Th, V2) - S(Th, V1))).simplify_full() == 0))
W = Qh - Qc
print("efficiency W/Q_h =", (W / Qh).simplify_full())
# check W as the area: int p dV around the cycle, adiabat p(V) from V T^3 = const
Vv = var("Vv")
Tad1 = Th * (V2 / Vv) ** (1 / 3); Tad2 = Th * (V1 / Vv) ** (1 / 3)
Warea = p(Th) * (V2 - V1) + integrate(p(Tad1), Vv, V2, V3) - p(Tc) * (V3 - V4) - integrate(p(Tad2), Vv, V1, V4)
print("int p dV around the cycle - (Q_h - Q_c) =", (Warea - W).canonicalize_radical().simplify_full())
