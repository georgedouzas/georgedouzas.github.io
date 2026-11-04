# Exercise exr-spd-six-count: zero modes on S^2 with a monopole of charge M, their chirality on S^2 (gamma_* = -sigma_3) and
# the four-dimensional chirality they pair with in a six-dimensional Weyl spinor of positive chirality
# (Gamma_* = gamma_5 x gamma_*^K: internal +1 pairs with right-handed, internal -1 with left-handed).
for M in range(-4, 5):
    n = abs(M)
    if M == 0:
        print("M =  0: no zero modes (vanishing theorem)"); continue
    ch = 1 if M > 0 else -1
    print("M = %+d: %d zero modes of internal chirality %+d -> %d %s-handed four-dimensional spinors, SU(2) spin %s" % (M, n, ch, n, "right" if ch > 0 else "left", QQ(n - 1) / 2))
