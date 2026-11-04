# Exercise exr-clm-symmetry-two-body-boost: the common boost delta x1 = delta x2 = t of two interacting particles.
t = var("t"); m1, m2 = var("m1 m2", domain="positive")
x1, x2 = function("x1")(t), function("x2")(t)
V = function("V")
X1, X2, V1, V2 = var("X1 X2 V1 V2")
L = m1 / 2 * V1**2 + m2 / 2 * V2**2 - V(X1 - X2)
dq = {X1: t, X2: t}; dv = {V1: 1, V2: 1}
dL = sum(diff(L, a) * dq[a] for a in (X1, X2)) + sum(diff(L, a) * dv[a] for a in (V1, V2))
on = {X1: x1, X2: x2, V1: diff(x1, t), V2: diff(x2, t)}
K = m1 * x1 + m2 * x2
print("delta L =", dL.subs(on).simplify_full(), "; delta L - dK/dt =", (dL.subs(on) - diff(K, t)).simplify_full())
Q = (diff(L, V1) * t + diff(L, V2) * t).subs(on) - K
print("Noether charge Q =", Q)
EL1 = diff(diff(L, V1).subs(on), t) - diff(L, X1).subs(on); EL2 = diff(diff(L, V2).subs(on), t) - diff(L, X2).subs(on)
dQ = diff(Q, t)
print("dQ/dt - t (EL1 + EL2) =", (dQ - t * (EL1 + EL2)).simplify_full(), "(so Q is conserved on the equations of motion)")
