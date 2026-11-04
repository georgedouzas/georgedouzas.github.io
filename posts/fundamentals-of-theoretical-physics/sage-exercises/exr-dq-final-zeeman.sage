# Exercise exr-dq-final-zeeman: Zeeman shifts g_J mu_B B m_j (GHz) of 2P_{1/2} and 2P_{3/2} for B = 0.1, 0.5, 1 T, and the field at which the
# 2P_{3/2} Zeeman spread equals the Dirac fine-structure interval 10.9437 GHz.
muB = 13.996245   # mu_B/h in GHz/T
for B in (0.1, 0.5, 1.0):
    p12 = [round(2/3 * muB * B * mj, 3) for mj in (-1/2, 1/2)]
    p32 = [round(4/3 * muB * B * mj, 3) for mj in (-3/2, -1/2, 1/2, 3/2)]
    print("B = %.1f T: 2P_1/2 shifts %s GHz; 2P_3/2 shifts %s GHz" % (B, p12, p32))
Bc = 10.9437 / (3 * 4/3 * muB)
print("2P_3/2 spread 3 g_J mu_B B equals 10.94 GHz at B = %.3f T" % Bc)
