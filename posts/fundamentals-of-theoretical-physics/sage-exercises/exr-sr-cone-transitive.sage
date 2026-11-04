# Exercise exr-sr-cone-transitive: a boost along z by log 5 followed by the rotation about y by arctan(3/4) maps (1,0,0,1)
# to (5,3,0,4).
K3 = matrix(SR, 4, 4); K3[0, 3] = K3[3, 0] = 1
Bz = (log(5)*K3).exp().simplify_full()
c, s = 4/5, 3/5
Ry = matrix(QQ, [[1, 0, 0, 0], [0, c, 0, s], [0, 0, 1, 0], [0, -s, 0, c]])     # exp(arctan(3/4) J2)
J2 = matrix(QQ, 4, 4); J2[1, 3] = 1; J2[3, 1] = -1
print("Ry = exp(arctan(3/4) J2):", (arctan(3/4)*J2.change_ring(SR)).exp().apply_map(lambda e: e.rectform().simplify_full()) == Ry)
L = Ry*Bz
print("boost along z by log 5:"); print(Bz)
print("Lambda k =", L*vector([1, 0, 0, 1]))
