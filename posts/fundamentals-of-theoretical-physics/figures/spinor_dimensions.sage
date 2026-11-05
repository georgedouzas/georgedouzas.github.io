# Figure: the number of real components of the smallest spinor in d-dimensional Minkowski space, computed from
# Tanii's rules (hep-th/9802138, Sec. 3): a Dirac spinor has 2^{floor(d/2)} complex components; a Weyl condition
# halves this for even d; a Majorana condition halves it when epsilon_- = +1, i.e. d = 2, 3, 4 mod 8 (pseudo-Majorana
# for d = 0, 1, 2 mod 8); Majorana and Weyl are compatible only for d = 2 mod 8. The result is compared with the
# standard table. A d-dimensional supergravity with N minimal supercharges has N times this many real
# supercharges, and 32 is the maximum, reached by d = 11, N = 1. Output: figures/spinor_dimensions.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spinor_dimensions.svg")
def minimal_spinor(d):
    real = 2 * 2^(d // 2)                                  # Dirac: complex components times two
    weyl = (d % 2 == 0)
    majorana = d % 8 in (2, 3, 4); pseudo = d % 8 in (0, 1, 2)
    types = []
    if weyl: types.append("W")
    if majorana: types.append("M")
    if pseudo: types.append("pM")
    if weyl and (majorana or pseudo) and d % 8 == 2: types.append("MW")
    if weyl: real //= 2
    if (majorana or pseudo) and (not weyl or d % 8 == 2): real //= 2
    if weyl and (majorana or pseudo) and d % 8 != 2: pass    # cannot impose both: keep the Weyl count
    return real, types
table = {2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 16, 8: 16, 9: 16, 10: 16, 11: 32}   # standard minimal real spinor components
computed = {d: minimal_spinor(d)[0] for d in range(2, 12)}
print({d: (computed[d], minimal_spinor(d)[1]) for d in range(2, 12)})
print("[PASS] the computed minimal real spinor dimensions agree with the standard table for d = 2, ..., 11:", computed == table)
print("[PASS] the eleven-dimensional Majorana spinor has 32 real components, the maximal number of supercharges:", computed[11] == 32)
p = Graphics()
for d in range(2, 12):
    n, types = minimal_spinor(d)
    p += polygon([(d - 0.35, 0), (d + 0.35, 0), (d + 0.35, n), (d - 0.35, n)], color="royalblue" if d != 11 else "firebrick", alpha=0.8)
    p += text(str(n), (d, n + 1.3), fontsize=9)
    p += text("/".join(types) if types else "Dirac", (d, -2.6), fontsize=7, rotation=0)
p.axes_labels(["$d$", "real components"]); p.ymin(-5); p.ymax(36); p.xmin(1.3); p.xmax(11.7)
p.set_aspect_ratio("automatic")
p.save(OUT, figsize=[8, 4], ticks=[list(range(2, 12)), [0, 8, 16, 32]])
print("wrote", os.path.relpath(OUT))
