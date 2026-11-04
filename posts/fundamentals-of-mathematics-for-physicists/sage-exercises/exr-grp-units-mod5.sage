# Exercise exr-grp-units-mod5: the non-zero residues modulo 5 form a group under multiplication (each has an
# inverse), while the non-zero residues modulo 6 do not; computed with Sage's Zmod.
for n in (5, 6):
    R = Zmod(n); X = [R(k) for k in range(1, n)]
    closed = all(a * b != 0 for a in X for b in X)
    inv = {int(a): [int(b) for b in X if a * b == 1] for a in X}
    print("n = %d: closed under multiplication: %s ; inverses: %s" % (n, closed, inv))
    print("        every element has an inverse:", all(len(v) == 1 for v in inv.values()))
R = Zmod(5)
print("multiplication table modulo 5 on {1,2,3,4}:")
for a in range(1, 5):
    print("   ", [int(R(a) * R(b)) for b in range(1, 5)])
print("2 * 3 = 0 modulo 6, so {1,...,5} is not closed:", Zmod(6)(2) * Zmod(6)(3) == 0)
print("Sage's group of units of Z_5:", [int(u) for u in Zmod(5).list_of_elements_of_multiplicative_group()], "; of Z_6:", [int(u) for u in Zmod(6).list_of_elements_of_multiplicative_group()])
