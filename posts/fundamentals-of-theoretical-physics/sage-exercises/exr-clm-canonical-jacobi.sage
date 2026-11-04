# Exercise exr-clm-canonical-jacobi: Jacobi identity and Leibniz rule for random polynomials in (q1, q2, p1, p2).
q1, q2, p1, p2 = var("q1 q2 p1 p2")
qs, ps = [q1, q2], [p1, p2]
def PB(f, g): return sum(diff(f, qs[i]) * diff(g, ps[i]) - diff(f, ps[i]) * diff(g, qs[i]) for i in range(2))
set_random_seed(1)
R = PolynomialRing(QQ, "q1,q2,p1,p2")
f, g, h = [SR(R.random_element(degree=3, terms=6)) for _ in range(3)]
print("f =", f); print("g =", g); print("h =", h)
print("Jacobi identity:", (PB(f, PB(g, h)) + PB(g, PB(h, f)) + PB(h, PB(f, g))).expand())
print("Leibniz rule {f, gh} - {f, g}h - g{f, h}:", (PB(f, g * h) - PB(f, g) * h - g * PB(f, h)).expand())
