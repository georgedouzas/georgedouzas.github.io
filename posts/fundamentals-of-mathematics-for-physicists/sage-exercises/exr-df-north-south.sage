# Exercise exr-df-north-south: the one-forms (1 - cos(theta)) dphi and -(1 + cos(theta)) dphi have the same exterior
# derivative, the area form, and differ by 2 dphi; the second, written in the north stereographic chart (which is
# centred at the south pole u = v = 0), has polynomial-over-positive components, smooth at the south pole.
S2 = Manifold(2, "S^2")
A = S2.open_subset("A"); U = S2.open_subset("U")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); ste = U.chart("u v")
th, ph = sph[:]; u, v = ste[:]
tr = sph.transition_map(ste, (sin(th)*cos(ph)/(1 - cos(th)), sin(th)*sin(ph)/(1 - cos(th))))
tr.set_inverse(2*atan(1/sqrt(u**2 + v**2)), atan2(v, u), check=False)
W = A.intersection(U); fs = sph.frame().restrict(W); fu = ste.frame().restrict(W)
eN = A.one_form(0, 1 - cos(th), name="eta_N"); eS = A.one_form(0, -1 - cos(th), name="eta_S")
area = A.diff_form(2, name="area"); area[0, 1] = sin(th)
print("d eta_N = area:", eN.exterior_derivative() == area, "; d eta_S = area:", eS.exterior_derivative() == area)
print("eta_N - eta_S =", (eN - eS).display())
eSW = eS.restrict(W)
print("eta_S in the stereographic chart:", eSW.display(fu, ste.restrict(W)))
c = [ci.expr() for ci in eSW[fu, :, ste.restrict(W)]]
print("components equal (2v, -2u)/(1 + u^2 + v^2):", bool((c[0] - 2*v/(1 + u**2 + v**2)).simplify_full() == 0 and (c[1] + 2*u/(1 + u**2 + v**2)).simplify_full() == 0))
