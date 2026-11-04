# Exercise exr-kkf-tower-count: number of circle modes with mass |n + alpha|/R < E for the periodic (alpha = 0) and
# antiperiodic (alpha = 1/2) spin structures, compared with 2 floor(ER) + 1 and 2 floor(ER + 1/2).
for ER in (QQ(3) / 10, QQ(12) / 10, QQ(27) / 10, QQ(104) / 10):
    nP = len([n for n in range(-100, 101) if abs(n) < ER])
    nA = len([n for n in range(-100, 101) if abs(n + QQ(1) / 2) < ER])
    print("ER = %s: periodic %d (formula %d), antiperiodic %d (formula %d)" % (ER, nP, 2 * floor(ER) + 1, nA, 2 * floor(ER + QQ(1) / 2)))
print("difference at most one for ER up to 50:", all(abs(len([n for n in range(-200, 201) if abs(n) < e]) - len([n for n in range(-200, 201) if abs(n + QQ(1) / 2) < e])) <= 1 for e in [QQ(k) / 7 for k in range(1, 350)]))
