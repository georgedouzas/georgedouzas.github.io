# Exercise exr-roots-polynomial-factor: factorisation of z^6 - 1 over QQ, over QQ(i) and over QQ(omega) with
# omega^2 + omega + 1 = 0; the number of linear factors depends on the field.
R.<z> = QQ[]
print("over QQ:        ", (z^6 - 1).factor())
Ki.<i> = QuadraticField(-1)
print("over QQ(i):     ", (z^6 - 1).change_ring(Ki).factor())
Kw.<w> = NumberField(z^2 + z + 1)
Fw = (z^6 - 1).change_ring(Kw).factor()
print("over QQ(omega): ", Fw)
print("number of linear factors over QQ, QQ(i), QQ(omega):", [sum(1 for f, m in F if f.degree() == 1) for F in ((z^6 - 1).factor(), (z^6 - 1).change_ring(Ki).factor(), Fw)])
