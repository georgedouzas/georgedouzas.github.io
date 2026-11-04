# Exercise exr-kkr-jets-inverse-derivative: the derivative of the inverse metric, d_r g^{mu nu} =
# -g^{mu a} (d_r g_{ab}) g^{b nu}, the rule the jet computation uses in place of differentiating the inverse. Checked
# for a symmetric 3 x 3 matrix of arbitrary functions of one variable.
s = var("s")
g = matrix(SR, 3, 3, lambda i, j: function("g%d%d" % (min(i, j), max(i, j)))(s))
gi = g.inverse()
lhs = gi.apply_map(lambda e: diff(e, s))
rhs = -gi * g.apply_map(lambda e: diff(e, s)) * gi
print("[PASS] d g^{-1} = -g^{-1} (d g) g^{-1} for a generic symmetric 3 x 3 metric:", bool((lhs - rhs).simplify_full() == 0))
