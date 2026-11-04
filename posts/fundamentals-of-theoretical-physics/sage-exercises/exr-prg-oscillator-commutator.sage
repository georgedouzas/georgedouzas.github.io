# Exercise exr-prg-oscillator-commutator: one mode with truncated ladder matrices: <0|q(t) q(0)|0> = e^{-i w t}/(2w),
# <0|[q(t), q(0)]|0> = -i sin(w t)/w, and G = sin(w t)/w has G'' + w^2 G = 0, G(0) = 0, G'(0) = 1.
N = 4; t = var("t", domain="real"); w = var("w", domain="positive")
a = matrix(SR, N, N, lambda i, j: sqrt(j) if j == i + 1 else 0); ad = a.transpose()
q = lambda s: (a * exp(-I * w * s) + ad * exp(I * w * s)) / sqrt(2 * w)
v0 = vector(SR, [1] + [0] * (N - 1))
W = (v0 * q(t) * q(0) * v0).simplify_full()
C = (v0 * (q(t) * q(0) - q(0) * q(t)) * v0)
print("<0|q(t)q(0)|0> =", W)
print("<0|[q(t), q(0)]|0> = -i sin(w t)/w:", bool((C - (-I * sin(w * t) / w)).exponentialize().simplify_full() == 0))
G = sin(w * t) / w
print("G'' + w^2 G =", (diff(G, t, 2) + w ** 2 * G).simplify_full(), "  G(0) =", G.subs(t=0), "  G'(0) =", diff(G, t).subs(t=0))
