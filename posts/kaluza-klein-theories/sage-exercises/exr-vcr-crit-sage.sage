# Exercise exr-vcr-crit-sage: the dilaton coupling of Cvetic-Lu-Pope, a^2 = 4 - 2(p-1)(D-p-1)/(D-2), required for a
# consistent S^n reduction retaining SO(n+1). Check its symmetry under p -> D - p (Hodge duality), tabulate it for p = 2, 3
# and D = 4, ..., 11, and find where it vanishes (no dilaton needed) for 2 <= p <= D - 2, D <= 26.
D, p = var("D p")
a2 = 4 - 2 * (p - 1) * (D - p - 1) / (D - 2)
print("a^2(D, p) - a^2(D, D - p) =", (a2 - a2.subs(p=D - p)).simplify_full())
for DD in range(4, 12):
    print("D = %2d: a^2(p=2) = %-6s a^2(p=3) = %s" % (DD, a2.subs(D=DD, p=2), a2.subs(D=DD, p=3)))
print("a^2 = 0 at (D, p):", [(DD, pp) for DD in range(4, 27) for pp in range(2, DD - 1) if a2.subs(D=DD, p=pp) == 0])
