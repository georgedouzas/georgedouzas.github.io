# Exercise exr-kkr-periodicity-check: the mode e^{i q y/R} returns to itself after y -> y + 2 pi R exactly when q is an
# integer; for q = p/r in lowest terms it returns after r turns, and for irrational q never.
q = var("q")
def phase(qq, turns=1):
    return exp(2 * pi * I * qq * turns)
print("phases after one turn for q = 0, 1, -2, 1/2, 1/3, 2/5:", [phase(x) for x in (0, 1, -2, 1/2, 1/3, 2/5)])
print("[PASS] one turn: periodic for integer q, not for q = 1/2, 1/3, 2/5:",
      all(bool(phase(x) == 1) for x in (0, 1, -2, 7)) and all(not bool(phase(x) == 1) for x in (1/2, 1/3, 2/5)))
print("[PASS] q = 2/5 returns after exactly 5 turns:", bool(phase(2/5, 5) == 1) and all(not bool(phase(2/5, k) == 1) for k in range(1, 5)))
print("[PASS] q = sqrt(2) never returns within 50 turns:", all(abs(N(phase(sqrt(2), k)) - 1) > 1e-6 for k in range(1, 51)))
