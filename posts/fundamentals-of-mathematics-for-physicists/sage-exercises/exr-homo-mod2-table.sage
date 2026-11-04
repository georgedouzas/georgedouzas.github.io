# Exercise exr-homo-mod2-table: reduction modulo 2 on Z_6 checked on all 36 pairs, its kernel and its fibres.
R6, R2 = Zmod(6), Zmod(2)
rho = lambda k: R2(int(k))
print("rho(j + k) = rho(j) + rho(k) for all 36 pairs:", all(rho(R6(j) + R6(k)) == rho(j) + rho(k) for j in range(6) for k in range(6)))
print("kernel:", [k for k in range(6) if rho(k) == 0], "; fibre over 1:", [k for k in range(6) if rho(k) == 1])
bad = [(j, k) for j in range(6) for k in range(6) if Zmod(4)(int(R6(j) + R6(k))) != Zmod(4)(j) + Zmod(4)(k)]
print("reduction modulo 4 fails on", len(bad), "pairs, e.g.", bad[:3], "(4 does not divide 6)")
