# Exercise exr-rn-integrate-out: tree-level matching of a heavy scalar Phi (mass M) coupled by -(g/2) Phi phi^2:
# Phi = -g phi^2/(2 M^2), L_eff = g^2 phi^4/(8 M^2), lambda_eff = -3 g^2/M^2.
Phi, phi, g, M, lam = var("Phi phi g M lam")
L = -M ** 2 * Phi ** 2 / 2 - g * Phi * phi ** 2 / 2
sol = solve(diff(L, Phi) == 0, Phi)[0].rhs()
Leff = L.subs(Phi=sol).simplify_full()
print("Phi =", sol, "   L_eff =", Leff)
print("lambda_eff from -lambda/4! phi^4 = L_eff:", solve(-lam / 24 * phi ** 4 == Leff, lam))
# first derivative correction: (box + M^2)^{-1} = M^{-2} (1 - box/M^2 + ...), so L_eff = (g^2/8) phi^2 (box + M^2)^{-1} phi^2
B = var("B")   # stands for the operator box acting on phi^2, treated as a number for the expansion
print("expansion of (g^2/8)/(B + M^2):", (g ** 2 / 8 / (B + M ** 2)).series(B, 2))
