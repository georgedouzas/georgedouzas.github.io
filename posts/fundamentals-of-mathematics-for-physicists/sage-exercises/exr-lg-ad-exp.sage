# Exercise exr-lg-ad-exp: Ad_{e^X} = e^{ad_X}: for X = t T_3 in su(2) the matrix of Ad_{e^{tT_3}} in the basis T_a
# is e^{t L_3} (the rotation by t), and for X = t H in sl(2, R) it is diag(1, e^{2t}, e^{-2t}) in the basis (H, E, F).
t = var("t", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
L3 = matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])
g = (t * T[2]).exp()
Ad = matrix(SR, 3, 3, lambda a, b: (-2 * (T[a] * g * T[b] * g.inverse()).trace()).exponentialize().simplify_full())
eL = (t * L3).exp().apply_map(lambda e: e.rectform().simplify_full())
print("Ad_{exp(t T_3)} =", Ad.apply_map(lambda e: e.trig_reduce()).list())
print("equals exp(t L_3):", bool((Ad - eL).apply_map(lambda e: e.exponentialize().simplify_full()) == 0))
H = matrix(SR, [[1, 0], [0, -1]]); E = matrix(SR, [[0, 1], [0, 0]]); F = matrix(SR, [[0, 0], [1, 0]])
h = (t * H).exp()
print("Ad_{exp(tH)}: E ->", (h * E * h.inverse()).simplify_full().list(), "; F ->", (h * F * h.inverse()).simplify_full().list(), "; H -> H:", (h * H * h.inverse()).simplify_full() == H)
print("e^{t ad_H} with ad_H = diag(0, 2, -2):", (t * diagonal_matrix(SR, [0, 2, -2])).exp().list())
