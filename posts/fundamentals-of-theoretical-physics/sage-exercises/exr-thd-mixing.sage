# Exercise exr-thd-mixing: entropy produced when two bodies of heat capacity C at T_1, T_2 equilibrate, and the entropy
# of mixing of two different ideal gases at equal T and p.
C, T1, T2, N, x = var("C T_1 T_2 N x"); assume(C > 0, T1 > 0, T2 > 0, N > 0, x > 0, x < 1)
Tf = (T1 + T2) / 2
dS = C * log(Tf / T1) + C * log(Tf / T2)
print("Delta S =", dS.simplify_log(), "; exp(Delta S/C) - 1 =", (exp(dS / C) - 1).simplify_full().factor())
print("numerical check T_1 = 300, T_2 = 400, C = 1:", dS.subs({C: 1, T1: 300, T2: 400}).n())
# mixing: N x particles of A in volume x V and N (1 - x) of B in (1 - x) V expand into V
dSmix = N * x * log(1 / x) + N * (1 - x) * log(1 / (1 - x))
print("Delta S_mix =", dSmix, "; at x = 1/2:", dSmix.subs(x=1 / 2).simplify_log(), "; dS/dx =", diff(dSmix, x).simplify_full(), "vanishes at x = 1/2:", bool(diff(dSmix, x).subs(x=1 / 2).simplify_full() == 0), "; d2S/dx2 =", diff(dSmix, x, 2).simplify_full())
