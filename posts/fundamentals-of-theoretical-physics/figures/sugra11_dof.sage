# Figure: physical degrees of freedom of eleven-dimensional supergravity and of its torus reduction to four
# dimensions (N = 8), counted with Tanii's light-cone formulas (hep-th/9802138, eq. 4.8): the graviton has
# (d-2)(d-1)/2 - 1, a p-form C(d-2, p), the gravitino (d-3) 2^{floor(d/2)}/2, a spin-1/2 field 2^{floor(d/2)}/2.
# In d = 11: 44 + 84 = 128 bosons and 128 fermions; in d = 4, N = 8: 2 + 56 + 70 = 128 and 16 + 112 = 128.
# Output: figures/sugra11_dof.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sugra11_dof.svg")
def graviton(d): return (d - 2) * (d - 1) // 2 - 1
def pform(d, p): return binomial(d - 2, p)
def gravitino(d): return (d - 3) * 2^(d // 2) // 2
def spinhalf(d): return 2^(d // 2) // 2
b11 = {"graviton $e_\\mu{}^a$": graviton(11), "3-form $B_{\\mu\\nu\\rho}$": pform(11, 3)}
f11 = {"gravitino $\\psi_\\mu$": gravitino(11)}
b4 = {"graviton": graviton(4), "28 vectors": 28 * pform(4, 1), "70 scalars": 70}
f4 = {"8 gravitini": 8 * gravitino(4), "56 spin-1/2": 56 * spinhalf(4)}
print("d = 11:", b11, f11); print("d = 4, N = 8:", b4, f4)
print("[PASS] eleven-dimensional supergravity has 44 + 84 = 128 bosonic and 128 fermionic degrees of freedom:",
      sum(b11.values()) == 128 and sum(f11.values()) == 128 and b11["graviton $e_\\mu{}^a$"] == 44 and b11["3-form $B_{\\mu\\nu\\rho}$"] == 84)
print("[PASS] the N = 8, d = 4 multiplet has 2 + 56 + 70 = 128 bosonic and 16 + 112 = 128 fermionic degrees of freedom:",
      sum(b4.values()) == 128 and sum(f4.values()) == 128)
def stack(items, x0, color, p):
    h = 0
    for name, n in items.items():
        p += polygon([(x0 - 0.3, h), (x0 + 0.3, h), (x0 + 0.3, h + n), (x0 - 0.3, h + n)], color=color, alpha=0.75, edgecolor="black")
        if n >= 10: p += text("%s: %d" % (name, n), (x0, h + n / 2), fontsize=8, color="white")
        else: p += text("%s: %d" % (name, n), (x0 + 0.33, h + n / 2), fontsize=7, color="black", horizontal_alignment="left")
        h += n
    return p
p = Graphics()
p = stack(b11, 1, "royalblue", p); p = stack(f11, 2, "firebrick", p); p = stack(b4, 3.5, "royalblue", p); p = stack(f4, 4.5, "firebrick", p)
for x0, lab in ((1, "bosons"), (2, "fermions"), (3.5, "bosons"), (4.5, "fermions")): p += text(lab, (x0, -10), fontsize=9)
p += text("$d = 11$, $N = 1$", (1.5, 140), fontsize=10); p += text("$d = 4$, $N = 8$ (torus reduction)", (4, 140), fontsize=10)
p += line([(0.5, 128), (5, 128)], color="gray", linestyle=":"); p += text("128", (0.35, 128), fontsize=8)
p.axes(False); p.ymin(-18); p.ymax(150); p.xmin(0.2); p.xmax(5.2)
p.set_aspect_ratio("automatic")
p.save(OUT, figsize=[8, 4.5])
print("wrote", os.path.relpath(OUT))
