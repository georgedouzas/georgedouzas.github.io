# Exercise exr-vcp-generations-chern: the Chern classes of the quintic to third order, c(TX) = (1 + J)^5/(1 + 5J) =
# 1 + 10 J^2 - 40 J^3, and the Chern numbers int c_2 J = 50 and int c_3 = -200 with int J^3 = 5. The Euler number of a
# free quotient by a group of order |G| is chi/|G|: for G = Z_5 x Z_5, chi = -8 and 4 net generations.
R.<J> = PowerSeriesRing(QQ, default_prec=5)
c = (1 + J) ** 5 / (1 + 5 * J)
print("c(TX) =", c.add_bigoh(4))
print("int c_2 J =", c[2] * 5, "; int c_3 =", c[3] * 5)
for order in [1, 5, 25]:
    chi = c[3] * 5 / order
    print("free quotient by a group of order", order, ": chi =", chi, ", net generations", abs(chi) / 2)
