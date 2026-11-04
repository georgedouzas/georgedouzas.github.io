# Exercise exr-maxwell-from-action: Maxwell's equations from the action -1/4 F^2 in 1+1 dimensions with Sage: the
# Euler-Lagrange equations of the components A_0, A_1, the gauge invariance of F, and the field strength of a
# plane wave potential.
t, x = var("t x", domain="real")
A0, A1, lam = [function(n)(t, x) for n in ["A0", "A1", "lam"]]
F01 = diff(A1, t) - diff(A0, x)                           # F_{01} = d_0 A_1 - d_1 A_0
eta = diagonal_matrix(SR, [1, -1])                       # signature (+, -)
L = -F01 * F01 * eta[0, 0] * eta[1, 1] / 2                # -1/4 F_{mu nu} F^{mu nu} = (1/2) F_{01}^2, the same in either signature
# Euler-Lagrange for A_nu: d_mu (dL/d(d_mu A_nu)) - dL/dA_nu = 0, computed with jet variables
a0, a1, a0t, a0x, a1t, a1x = var("a0 a1 a0t a0x a1t a1x")
Lj = L.subs({diff(A1, t): a1t, diff(A1, x): a1x, diff(A0, t): a0t, diff(A0, x): a0x}).subs({A0: a0, A1: a1})
back = {a1t: diff(A1, t), a1x: diff(A1, x), a0t: diff(A0, t), a0x: diff(A0, x), a0: A0, a1: A1}
EL0 = (diff(Lj, a0t).subs(back).diff(t) + diff(Lj, a0x).subs(back).diff(x) - diff(Lj, a0).subs(back)).simplify_full()
EL1 = (diff(Lj, a1t).subs(back).diff(t) + diff(Lj, a1x).subs(back).diff(x) - diff(Lj, a1).subs(back)).simplify_full()
print("equation for A_0:", EL0, "= 0  (Gauss's law, d_x F_{10} = 0)")
print("equation for A_1:", EL1, "= 0  (d_t F_{01} = 0)")
F_gauge = (diff(A1 + diff(lam, x), t) - diff(A0 + diff(lam, t), x)).simplify_full()
print("F is gauge invariant under A -> A + d lambda:", bool((F_gauge - F01).simplify_full() == 0))
E = function("E")(t, x)
print("both equations say that F_01 is constant: d_x F_01 = 0 and d_t F_01 = 0; in 1+1 dimensions there are no propagating")
print("electromagnetic waves, only a constant electric field (the photon has no transverse directions there).")
print("EL_0 = -d_x F_{01} and EL_1 = d_t F_{01} as expressions:", bool((EL0 + diff(F01, x)).simplify_full() == 0), bool((EL1 - diff(F01, t)).simplify_full() == 0))
