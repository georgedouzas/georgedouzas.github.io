# Exercise exr-hopf-chern: the first Chern number of the Hopf bundle. The potential cos(theta) dphi belongs to the
# connection sigma^3 = dpsi + cos(theta) dphi, whose fibre angle psi has period 4 pi on SU(2); normalised to the
# 2 pi-periodic angle psi/2 the potential is (1/2) cos(theta) dphi, and (1/2 pi) times the integral of its curvature
# is c_1 = -1. Treating psi as 2 pi-periodic instead gives the SO(3) bundle with c_1 = -2.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
Apsi = A.one_form(0, cos(th), name="A_psi")                  # potential of sigma^3, fibre angle psi (period 4 pi)
Achi = A.one_form(0, cos(th) / 2, name="A_chi")              # normalised to chi = psi/2 (period 2 pi)
for name, form in [("A_psi = cos(theta) dphi", Apsi), ("A_chi = (1/2) cos(theta) dphi", Achi)]:
    F = form.exterior_derivative()
    total = integrate(integrate(F[0, 1].expr(), ph, 0, 2*pi), th, 0, pi)
    print(name, ": F = dA has component", F[0, 1].expr(), "; integral over S^2 =", total)
Fchi = Achi.exterior_derivative()
c1 = integrate(integrate(Fchi[0, 1].expr(), ph, 0, 2*pi), th, 0, pi) / (2*pi)
print("Hopf bundle S^3 -> S^2: c_1 = (1/2 pi) integral of d A_chi =", c1, "; an integer:", c1 in ZZ, "; equals -1:", bool(c1 == -1))
c1_so3 = integrate(integrate(Apsi.exterior_derivative()[0, 1].expr(), ph, 0, 2*pi), th, 0, pi) / (2*pi)
print("treating psi as 2 pi-periodic (the quotient SO(3) -> S^2): c_1 =", c1_so3)
