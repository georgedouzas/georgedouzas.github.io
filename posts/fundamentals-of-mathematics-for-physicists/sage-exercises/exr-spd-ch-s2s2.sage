# Exercise exr-spd-ch-s2s2: ch = exp(M1 w1 + M2 w2) on S^2 x S^2 with w1^2 = w2^2 = 0, int w1 w2 = 1: int ch_2 = M1 M2.
R.<M1, M2, w1, w2> = QQ[]
J = R.ideal([w1 ^ 2, w2 ^ 2])
u = M1 * w1 + M2 * w2
ch = J.reduce(sum(u ^ k / factorial(k) for k in range(4)))
print("ch =", ch)
print("int ch_2 = coefficient of w1 w2:", ch.coefficient({w1: 1, w2: 1}))
