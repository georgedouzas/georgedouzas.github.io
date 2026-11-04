# Exercise exr-fbun-su2-gauge-rotation: under g = e^{alpha(x) T_3} the constant potential A_x = c T_1 becomes
# g^{-1} A_x g + g^{-1} d_x g = c (cos(alpha) T_1 - sin(alpha) T_2) + alpha'(x) T_3 (with T_a = -i sigma_a / 2):
# the adjoint action rotates the components and the derivative term adds a T_3 component.
x, c = var("x c", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
alpha = function("alpha")(x)
g = matrix(SR, [[exp(-I * alpha / 2), 0], [0, exp(I * alpha / 2)]])   # e^{alpha T_3}
print("g = e^{alpha T_3}:", bool((g - (alpha * T[2]).exp()).simplify_full() == 0))
Ax = c * T[0]
Ap = (g.inverse() * Ax * g + g.inverse() * diff(g, x)).simplify_full()
def comps(M):  # components in the basis T_a: M = sum m_a T_a, m_a = 2 i tr(... ) from tr(T_a T_b) = -delta/2
    return [(-2 * (M * T[a]).trace()).simplify_full() for a in range(3)]
print("components of the transformed A_x in (T_1, T_2, T_3):", [e.expand().maxima_methods().demoivre().simplify_full() for e in comps(Ap)])
print("= (c cos(alpha), -c sin(alpha), alpha'):", bool(all((u - w).exponentialize().expand().simplify_full() == 0 for u, w in
      zip(comps(Ap), [c * cos(alpha), -c * sin(alpha), diff(alpha, x)]))))
