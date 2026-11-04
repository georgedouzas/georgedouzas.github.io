# Exercise exr-fbun-dirac-gauge: the Dirac potentials A_+ = (cos(theta) - 1) d phi (regular at the north pole) and
# A_- = (cos(theta) + 1) d phi (regular at the south pole) differ by A_+ - A_- = -2 d phi, the gauge transformation
# of the transition function e^{-2 i phi} (in the convention A -> A + d alpha), and they have the same curvature.
S2 = Manifold(2, "S2"); U = S2.open_subset("U")
sph = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
Ap = U.one_form(0, cos(th) - 1, name="A_+"); Am = U.one_form(0, cos(th) + 1, name="A_-")
alpha = U.scalar_field(-2 * ph, name="alpha")
print("A_+ - A_- =", (Ap - Am).display())
print("A_+ = A_- + d alpha with alpha = -2 phi:", Ap == Am + alpha.differential())
print("dA_+ =", Ap.exterior_derivative().display(), "; dA_- =", Am.exterior_derivative().display())
print("same curvature:", Ap.exterior_derivative() == Am.exterior_derivative())
# regularity: the component A_phi divided by sin(theta) is the length of A in an orthonormal frame
for nm, Aphi in [("A_+", cos(th) - 1), ("A_-", cos(th) + 1)]:
    print(f"|{nm}| = |A_phi|/sin(theta) near theta = 0 and pi:", limit(Aphi / sin(th), th=0, dir="+"), limit(Aphi / sin(th), th=pi, dir="-"))
