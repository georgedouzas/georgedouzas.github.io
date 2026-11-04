# Exercise exr-qm-commutator-xp2: [x, p^2] = 2 i hbar p, [x^2, p] = 2 i hbar x, [p, V] = -i hbar V' on a symbolic f.
x = var("x"); hbar = var("hbar"); f = function("f")(x); V = function("V")(x)
P = lambda g: -I * hbar * diff(g, x); X = lambda g: x * g
print("[x, p^2] f - 2 i hbar p f =", (X(P(P(f))) - P(P(X(f))) - 2 * I * hbar * P(f)).simplify_full())
print("[x^2, p] f - 2 i hbar x f =", (X(X(P(f))) - P(X(X(f))) - 2 * I * hbar * x * f).simplify_full())
print("[p, V] f + i hbar V' f =", (P(V * f) - V * P(f) + I * hbar * diff(V, x) * f).simplify_full())
