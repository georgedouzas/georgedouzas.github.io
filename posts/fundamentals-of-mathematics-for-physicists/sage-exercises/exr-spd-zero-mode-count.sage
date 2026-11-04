# Exercise exr-spd-zero-mode-count: for M = 3, the half-integers m for which the separated zero-mode solutions on the unit
# sphere are admissible. Lower component g = sin^a(t/2) cos^b(t/2) with a = -m - 1/2, b = m + M - 1/2; upper component
# f = sin^a(t/2) cos^b(t/2) with a = m - 1/2, b = -m - M - 1/2. Near t = 0, |g|^2 sin t ~ t^{2a+1}; near t = pi,
# ~ (pi - t)^{2b+1}: normalisable iff a > -1 and b > -1, bounded iff a >= 0 and b >= 0.
M = 3
ms = [QQ(k) / 2 for k in range(-9, 9, 2)]
def status(a, b):
    return ("bounded" if a >= 0 and b >= 0 else ("normalisable" if a > -1 and b > -1 else "excluded"))
lower = {m: status(-m - 1/2, m + M - 1/2) for m in ms}
upper = {m: status(m - 1/2, -m - M - 1/2) for m in ms}
for m in ms:
    print("m = %5s: lower component %-12s upper component %s" % (m, lower[m], upper[m]))
nl = sum(1 for m in ms if lower[m] == "bounded"); nu = sum(1 for m in ms if upper[m] == "bounded")
print("zero modes: %d of chirality +1 (lower), %d of chirality -1 (upper); index = %d" % (nl, nu, nl - nu))
