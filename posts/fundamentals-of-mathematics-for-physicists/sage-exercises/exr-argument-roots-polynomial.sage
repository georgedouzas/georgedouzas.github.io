# Exercise exr-argument-roots-polynomial: the four solutions of z^4 = -16 as algebraic numbers and in polar form.
z = var("z")
roots = [r for r, m in (z^4 + 16).roots(ring=QQbar)]
print("roots of z^4 + 16:", roots)
print("all have modulus 2:", all(abs(r) == 2 for r in roots))
polar = [2*exp(I*pi*(2*k + 1)/4) for k in range(4)]
print("polar form 2 e^{i pi (2k+1)/4}, k = 0..3, in Cartesian form:", [p.simplify_full() for p in polar])
print("they solve z^4 = -16:", all(bool((p^4).simplify_full() == -16) for p in polar))
