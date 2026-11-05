# Figure: the Hodge diamonds of two Calabi-Yau threefolds, the quintic hypersurface in CP^4 (h^{1,1} = 1,
# h^{2,1} = 101) and the resolved orbifold T^6/Z_3 (h^{1,1} = 36, h^{2,1} = 0), with their Euler numbers
# chi = 2(h^{1,1} - h^{2,1}) and net generation numbers |h^{1,1} - h^{2,1}|. Checks: the Chern classes of a degree-t
# hypersurface in CP^4 from c(TX) = (1 + J)^5/(1 + tJ), so c_1 = 0 exactly for t = 5; c_3 = -40 J^3 and
# int J^3 = 5 give chi = -200; the count h^{2,1} = 126 - 25 = 101 of complex-structure deformations; the alternating
# sum of the diamond; the quartic K3 surface (chi = 24); and the Euler numbers chi(G/H) = |W_G|/|W_H| of the three
# six-dimensional nearly-Kahler cosets of equal rank (2, 4, 6). Output: figures/hodge_diamond.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hodge_diamond.svg")


def diamond(h11, h21):
    """Hodge numbers h^{p,q} of a Calabi-Yau threefold (full SU(3) holonomy)."""
    h = {}
    for p in range(4):
        for q in range(4):
            h[(p, q)] = 0
    for (p, q) in [(0, 0), (3, 0), (0, 3), (3, 3)]:
        h[(p, q)] = 1
    for (p, q) in [(1, 1), (2, 2)]:
        h[(p, q)] = h11
    for (p, q) in [(2, 1), (1, 2)]:
        h[(p, q)] = h21
    return h


def euler(h):
    return sum((-1) ** (p + q) * h[(p, q)] for (p, q) in h)


def draw(h, title, col):
    G = Graphics()
    for (p, q), v in h.items():
        X, Y = q - p, 6 - (p + q)          # rows by p + q (top: h^{0,0}), columns by q - p
        G += text(str(v), (X, Y), fontsize=15 if v else 12, color=(col if v not in (0, 1) else "black"))
    G += text(title, (0, 7.2), fontsize=12, color="black")
    G.axes(False); G.set_aspect_ratio(1)
    return G


# Chern classes of a degree-t hypersurface in CP^4: c(TX) = (1 + J)^5 / (1 + t J) truncated at J^3
R.<J> = PowerSeriesRing(QQ, default_prec=5)
t = var("t")
def chern(tt):
    c = (1 + J) ** 5 / (1 + tt * J)
    return [c[i] for i in range(4)]
print("[PASS] degree-t hypersurface in CP^4: c_1 = (5 - t) J, so c_1 = 0 (Calabi-Yau) exactly for t = 5:",
      all(chern(tt)[1] == 5 - tt for tt in range(1, 9)) and [tt for tt in range(1, 9) if chern(tt)[1] == 0] == [5])
c = chern(5)
print("[PASS] quintic: c(TX) = 1 + 0 J + 10 J^2 - 40 J^3, so c_3 = -40 J^3:", c == [1, 0, 10, -40])
chi_quintic = c[3] * 5                     # int_X J^3 = degree = 5
print("[PASS] quintic: chi = int c_3 = -40 * 5 = -200:", chi_quintic == -200)
monomials = binomial(5 + 5 - 1, 5)
print("[PASS] quintic: h^{2,1} = 126 - 25 = 101 (degree-5 monomials in 5 variables minus dim GL(5)):", monomials == 126 and monomials - 25 == 101)
hq = diamond(1, 101)
print("[PASS] quintic: the alternating sum of the Hodge diamond is chi = 2(h^{1,1} - h^{2,1}) = -200:", euler(hq) == -200 == 2 * (1 - 101))
print("[PASS] quintic: net generations |h^{1,1} - h^{2,1}| = |chi|/2 = 100:", abs(1 - 101) == abs(euler(hq)) / 2 == 100)
ho = diamond(36, 0)
print("[PASS] T^6/Z_3 (resolved): chi = 2(36 - 0) = 72 and 36 net generations:", euler(ho) == 72 and abs(36 - 0) == 36)
# the quartic surface in CP^3: c(TX) = (1 + J)^4/(1 + 4J), c_2 = 6 J^2, int J^2 = 4
c4 = (1 + J) ** 4 / (1 + 4 * J)
print("[PASS] quartic in CP^3 (K3): c_1 = 0, c_2 = 6 J^2, chi = 6 * 4 = 24:", c4[1] == 0 and c4[2] == 6 and c4[2] * 4 == 24)
# Euler numbers of equal-rank cosets G/H: chi = |W_G| / |W_H|
WG2, WA2, WC2, WA1 = [WeylGroup(ct).order() for ct in (["G", 2], ["A", 2], ["C", 2], ["A", 1])]
chis = (WG2 // WA2, WC2 // (WA1 * 1), WA2 // 1)
print("[PASS] nearly-Kahler cosets: chi(G_2/SU(3)) = 12/6 = 2, chi(Sp(4)/(SU(2) x U(1))) = 8/2 = 4, chi(SU(3)/U(1)^2) = 6/1 = 6:",
      (WG2, WA2, WC2, WA1) == (12, 6, 8, 2) and chis == (2, 4, 6))
print("[PASS] nearly-Kahler cosets: chi/2 = 1, 2, 3 generations:", tuple(x // 2 for x in chis) == (1, 2, 3))

g = graphics_array([draw(hq, "quintic in $\\mathbb{CP}^4$:  $\\chi = -200$", "firebrick"),
                    draw(ho, "$T^6/\\mathbb{Z}_3$ resolved:  $\\chi = 72$", "royalblue")])
fig = g.matplotlib(figsize=[9, 4.6]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
