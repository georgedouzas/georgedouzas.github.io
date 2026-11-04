# Exercise exr-tsb-fundamental (Bricmont, Exercise 5.12): a system with E = a V T^4 and P = E/(3V); its fundamental
# relation S(E, V) from dS = dE/T + P dV/T, checked by differentiation, and the adiabats.
E, V, a, T = var('E V a T'); assume(E > 0, V > 0, a > 0, T > 0)
Tf = (E / (a * V))^(1 / 4); P = E / (3 * V)
S = 4 / 3 * a^(1 / 4) * V^(1 / 4) * E^(3 / 4)
print("dS/dE = 1/T:", bool((diff(S, E) - 1 / Tf).canonicalize_radical() == 0))
print("dS/dV = P/T:", bool((diff(S, V) - P / Tf).canonicalize_radical() == 0))
print("S is homogeneous of degree one: S(2E, 2V) = 2 S(E, V):", bool((S.subs(E=2 * E, V=2 * V) - 2 * S).simplify_full() == 0))
ST = S.subs(E=a * V * T^4).canonicalize_radical()
print("S(T, V) =", ST, "  so the adiabats are V T^3 = const and P V^(4/3) = const")
sv = var('sv'); assume(sv > 0)
Es = (3 * sv / (4 * (a * V)^(1 / 4)))^(4 / 3)
print("E(S, V) inverts S(E, V):", bool((S.subs(E=Es) - sv).canonicalize_radical() == 0))
print("P V^(4/3) on the adiabat S = sv:", (P * V^(4 / 3)).subs(E=Es).canonicalize_radical(), "(independent of V)")
