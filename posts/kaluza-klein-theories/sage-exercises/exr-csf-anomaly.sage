# Exercise exr-csf-anomaly: abelian anomaly sums over the left-handed four-dimensional fermions of the T_3 model, with the
# charges (t, y) of U(1)_3 x U(1)_8; right-handed fermions enter with opposite charges.
def sums(left):
    t = [c[0] for c in left]; y = [c[1] for c in left]
    return {"t^3": sum(a ** 3 for a in t), "y^3": sum(b ** 3 for b in y), "t^2 y": sum(a ** 2 * b for a, b in left),
            "t y^2": sum(a * b ** 2 for a, b in left), "t": sum(t), "y": sum(y)}
h = QQ(1) / 2
fund_R = [(h, 1)]; fund_L = [(-h, 1)]
adj_R = [(h, 3), (h, -3)]; adj_L = [(-h, -3), (-h, 3)]
for name, R_, L_ in (("3", fund_R, fund_L), ("8", adj_R, adj_L)):
    left = L_ + [(-a, -b) for a, b in R_]
    print("%s: left-handed charges %s" % (name, left)); print("   anomaly sums:", sums(left))
