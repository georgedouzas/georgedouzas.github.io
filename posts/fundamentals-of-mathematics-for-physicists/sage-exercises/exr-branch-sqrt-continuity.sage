# Exercise exr-branch-sqrt-continuity: jumps of the principal square root and logarithm across the negative real
# axis, and continuity across the positive real axis.
eps = 1e-6
for x0 in (-4, 4):
    above, below = CC(x0, eps), CC(x0, -eps)
    print("x0 =", x0, ": sqrt above - sqrt below =", above.sqrt() - below.sqrt(), "  Log above - Log below =", above.log() - below.log())
print("jump of sqrt across the cut is 4i (to 1e-5):", abs(CC(-4, eps).sqrt() - CC(-4, -eps).sqrt() - 4*I) < 1e-5)
print("jump of Log across the cut is 2 pi i (to 1e-5):", abs(CC(-4, eps).log() - CC(-4, -eps).log() - 2*pi*I) < 1e-5)
print("no jump across the positive axis (to 1e-5):", abs(CC(4, eps).sqrt() - CC(4, -eps).sqrt()) < 1e-5 and abs(CC(4, eps).log() - CC(4, -eps).log()) < 1e-5)
