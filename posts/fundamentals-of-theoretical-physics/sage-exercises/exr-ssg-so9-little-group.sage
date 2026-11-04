# Exercise exr-ssg-so9-little-group: the massless states of eleven-dimensional supergravity carry representations of
# the little group SO(9): the graviton is the symmetric traceless tensor (44), the three-form the third exterior power
# of the vector (84), the gravitino the gamma-traceless vector-spinor (128 = 9 x 16 - 16). Dimensions from Sage's Weyl
# character ring of B4 (fundamental weights: vector w1, spinor w4).
B4 = WeylCharacterRing("B4", style="coroots"); w = B4.fundamental_weights()
vec, spin = B4(w[1]), B4(w[4])
grav, form3, gravitino = B4(2 * w[1]), B4(w[3]), B4(w[1] + w[4])
print("dims: vector", vec.degree(), "spinor", spin.degree(), "graviton", grav.degree(), "3-form", form3.degree(), "gravitino", gravitino.degree())
print("[PASS] graviton 44 = 9*10/2 - 1, three-form 84 = binomial(9, 3), gravitino 128 = 9*16 - 16:",
      grav.degree() == 44 and form3.degree() == 84 and gravitino.degree() == 128 and 9 * 16 - 16 == 128)
print("[PASS] vector x spinor = gravitino + spinor (the gamma-trace removes one spinor):", vec * spin == gravitino + spin)
print("Sym^2(vector) =", vec.symmetric_power(2), "; Lambda^3(vector) =", vec.exterior_power(3))
print("[PASS] Sym^2(vector) = graviton + singlet and Lambda^3(vector) = 3-form: 45 = 44 + 1, 84:",
      vec.symmetric_power(2) == grav + B4(0, 0, 0, 0) and vec.exterior_power(3) == form3)
print("[PASS] bosons 44 + 84 = 128 = fermions:", grav.degree() + form3.degree() == gravitino.degree())
