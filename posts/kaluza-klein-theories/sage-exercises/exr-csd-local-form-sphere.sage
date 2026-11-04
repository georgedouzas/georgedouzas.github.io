# Exercise exr-csd-local-form-sphere: the Maurer-Cartan form of the coset representative s = exp(ph Q_3) exp(th Q_2)
# of S^2 = SU(2)/U(1), Q_a = sigma_a/(2i), split into the isotropy part (along Q_3) and the frame (along Q_1, Q_2).
th, ph, dth, dph = var("th ph dth dph")
Q = [matrix(SR, [[0, 1], [1, 0]]) / (2 * I), matrix(SR, [[0, -I], [I, 0]]) / (2 * I), matrix(SR, [[1, 0], [0, -1]]) / (2 * I)]
def expQ(k, t): return (t * Q[k]).exp().simplify_full()
s = expQ(2, ph) * expQ(1, th)
ds = diff(s, th) * dth + diff(s, ph) * dph
MC = (s.inverse() * ds).simplify_full()
# components along Q_a: X = sum c_a Q_a with c_a = -2 Tr(X Q_a)
c = [(-2 * (MC * Q[a]).trace()).maxima_methods().demoivre().simplify_full() for a in range(3)]   # demoivre: exponentials of i th to cos, sin
print("s^{-1} ds = theta^1 Q_1 + theta^2 Q_2 + theta^3 Q_3 with theta =", c)
print("[PASS] theta^1 = -sin(th) dph, theta^2 = dth (the frame), theta^3 = cos(th) dph (the R-connection):",
      bool(c[0] == -sin(th) * dph) and bool(c[1] == dth) and bool(c[2] == cos(th) * dph))
print("[PASS] the frame gives the round metric (theta^1)^2 + (theta^2)^2 = dth^2 + sin(th)^2 dph^2:", bool((c[0]^2 + c[1]^2 - dth^2 - sin(th)^2 * dph^2).expand() == 0))
