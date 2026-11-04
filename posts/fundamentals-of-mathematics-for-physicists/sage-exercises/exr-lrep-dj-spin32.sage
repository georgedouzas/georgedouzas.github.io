# Exercise exr-lrep-dj-spin32: the state |j, j> rotated about the first axis, d^j(theta) = exp(-i theta J_1), for j = 3/2
# and j = 2: the probabilities of the weights m are binomial(2j, j - m) cos^{2(j+m)}(theta/2) sin^{2(j-m)}(theta/2), the
# amplitudes have period 4 pi for half-integer j, and the probabilities have period 2 pi.
th = var("th", domain="real")
def spin_matrices(j):
    ms = [j - k for k in range(2 * j + 1)]; n = len(ms)
    J3 = diagonal_matrix(SR, ms)
    Jp = matrix(SR, n, n, lambda r, c: sqrt(j * (j + 1) - ms[c] * (ms[c] + 1)) if r == c - 1 else 0)
    return ms, J3, Jp, Jp.transpose()
for j in [QQ(3) / 2, 2]:
    ms, J3, Jp, Jm = spin_matrices(j)
    J1 = (Jp + Jm) / 2
    d = (-I * th * J1).exp().apply_map(lambda e: e.rectform().simplify_full().trig_reduce())
    amp = d.column(0)                                # the rotated state |j, j>, components m = j, ..., -j
    prob = [(a * a.conjugate()).simplify_full().trig_reduce() for a in amp]
    expected = [binomial(2 * j, j - m) * cos(th / 2)^(2 * (j + m)) * sin(th / 2)^(2 * (j - m)) for m in ms]
    print(f"j = {j}: amplitudes <m|d(theta)|j>, m = {ms}:")
    for m, a in zip(ms, amp): print("   ", m, ":", a.trig_simplify())
    print("   probabilities = binomial(2j, j-m) cos^{2(j+m)} sin^{2(j-m)}:", all(bool((p - e).trig_reduce().simplify_full() == 0) for p, e in zip(prob, expected)),
          "; they sum to 1:", bool(sum(prob).simplify_full() == 1))
    print("    amplitude at theta = 2 pi:", [a.subs(th=2 * pi) for a in amp], "; at 4 pi:", [a.subs(th=4 * pi) for a in amp])
    print("    probabilities at theta = pi/2:", [p.subs(th=pi / 2) for p in prob], "; at pi:", [p.subs(th=pi) for p in prob])
