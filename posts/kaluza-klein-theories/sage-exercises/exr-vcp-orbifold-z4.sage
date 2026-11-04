# Exercise exr-vcp-orbifold-z4: fixed points of z -> i z on the square torus C/(Z + iZ). In lattice coordinates the
# rotation is M = [[0, -1], [1, 0]]; points fixed modulo the lattice solve (M - 1) v in Z^2 and number det(1 - M) = 2:
# z = 0 and z = (1 + i)/2. The square M^2 = -1 (z -> -z) has det(1 - M^2) = 4 fixed points, 0, 1/2, i/2, (1 + i)/2;
# z -> i z exchanges 1/2 and i/2. The Z_4 orbifold T^2/Z_4 is a sphere with cone points of angles 2pi/4, 2pi/4, 2pi/2.
M = matrix(ZZ, [[0, -1], [1, 0]])
for k, name in [(1, "z -> i z"), (2, "z -> -z"), (3, "z -> -i z")]:
    Mk = M ** k
    pts = [vector(QQ, [a / 4, b / 4]) for a in range(4) for b in range(4)]
    fixed = [tuple(v) for v in pts if all(x in ZZ for x in (Mk - 1) * v)]
    print(name, ": det(1 - M^k) =", (1 - Mk).det(), "; fixed points (lattice coordinates):", fixed)
# orbit of the half-period points under z -> i z
print("i * (1/2) = i/2: the points 1/2 and i/2 form one orbit of length 2:", M * vector(QQ, [1/2, 0]) == vector(QQ, [0, 1/2]))
# Riemann-Hurwitz: chi(T^2) = 0 = 4 (chi(S) - sum over cone points (1 - 1/m_i)) with m = (4, 4, 2)
chiS = sum(1 - 1 / QQ(m) for m in [4, 4, 2])
print("Riemann-Hurwitz: chi(quotient) = chi(T^2)/4 + sum (1 - 1/m_i) =", chiS, "= 2, a sphere:", chiS == 2)
