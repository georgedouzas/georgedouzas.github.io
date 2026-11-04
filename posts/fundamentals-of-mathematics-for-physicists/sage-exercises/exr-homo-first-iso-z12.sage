# Exercise exr-homo-first-iso-z12: reduction modulo 4, Z_12 -> Z_4: its kernel, its fibres (the cosets of the
# kernel) and the induced isomorphism Z_12/<4> -> Z_4.
R12, R4 = Zmod(12), Zmod(4)
rho = lambda k: R4(k)
print("well defined and a homomorphism:", all(rho((j + k) % 12) == rho(j) + rho(k) for j in range(12) for k in range(12)))
ker = [k for k in range(12) if rho(k) == 0]
print("kernel:", ker)
fibres = {int(r): [k for k in range(12) if rho(k) == r] for r in R4}
print("fibres:", fibres)
cosets = {tuple(sorted(int(R12(a) + R12(h)) for h in ker)) for a in range(12)}
print("the fibres are the cosets of the kernel:", set(tuple(f) for f in fibres.values()) == cosets)
print("the induced map on cosets is a bijection onto Z_4:", len(fibres) == 4 and all(len(f) == 3 for f in fibres.values()))
