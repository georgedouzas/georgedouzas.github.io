# Exercise exr-tf-contraction-pairs: the (1, 2) tensor field S = d_theta (x) dtheta (x) dphi on the sphere contracted
# on its upper index with the first and with the second lower index: the two contractions differ.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
S = A.tensor_field(1, 2, name="S"); S[0, 0, 1] = 1                    # S^theta_{theta phi} = 1
print("S =", S.display())
c1 = S.trace(0, 1); c2 = S.trace(0, 2)
print("contraction S^i_{i k} =", c1.display())
print("contraction S^i_{j i} =", c2.display())
