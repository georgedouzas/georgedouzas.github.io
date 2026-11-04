# Exercise exr-of-dual-frame: the dual coframe of the orthonormal frame e1 = d_theta, e2 = d_phi/sin(theta) of the
# sphere is (dtheta, sin(theta) dphi), with SageManifolds' vector_frame and coframe.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
e = A.vector_frame(("e1", "e2"), [A.vector_field(1, 0), A.vector_field(0, 1/sin(th))], symbol_dual=("e^1", "e^2"))
ec = e.coframe()
print(ec[0].display(sph.frame()), ";", ec[1].display(sph.frame()))
print("e^1 = dtheta and e^2 = sin(theta) dphi:", ec[0] == sph.coframe()[0], ec[1] == sin(th)*sph.coframe()[1])
print("e^a(e_b) = delta^a_b:", [[ec[a](e[b]).expr() for b in range(2)] for a in range(2)])
