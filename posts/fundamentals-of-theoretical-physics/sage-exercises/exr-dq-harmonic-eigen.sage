# Exercise exr-dq-harmonic-eigen: the spinor harmonics Omega_{j l m} for l = 1 are eigenfunctions of sigma.L = sigma_3 L_z +
# (sigma_+ L_- + sigma_- L_+)/2 with eigenvalue l = 1 for j = 3/2 and -(l + 1) = -2 for j = 1/2.
th, ph = var("theta phi", domain="real")
Y = lambda l, m: spherical_harmonic(l, m, th, ph) if abs(m) <= l else 0
Lz = lambda f: -I * diff(f, ph)
Lp = lambda f: exp(I * ph) * (diff(f, th) + I * cot(th) * diff(f, ph))
Lm = lambda f: exp(-I * ph) * (-diff(f, th) + I * cot(th) * diff(f, ph))
def sigL(O):   # sigma.L on (O1, O2), with sigma_+ = sigma_1 + i sigma_2 raising: (sigma_+ L_-)(O1, O2) = (2 L_- O2, 0)
    return [Lz(O[0]) + Lm(O[1]), -Lz(O[1]) + Lp(O[0])]
def Omega(j, l, m):
    if j == l + 1/2:
        return [sqrt(l + m + 1/2) * Y(l, m - 1/2) / sqrt(2 * l + 1), sqrt(l - m + 1/2) * Y(l, m + 1/2) / sqrt(2 * l + 1)]
    return [-sqrt(l - m + 1/2) * Y(l, m - 1/2) / sqrt(2 * l + 1), sqrt(l + m + 1/2) * Y(l, m + 1/2) / sqrt(2 * l + 1)]
l = 1
for j, ev in [(QQ(3) / 2, 1), (QQ(1) / 2, -2)]:
    for m in [QQ(k) / 2 for k in range(-int(2 * j), int(2 * j) + 1, 2)]:
        O = Omega(j, l, m); S = sigL(O)
        ok = all(bool((a - ev * b).simplify_full() == 0) for a, b in zip(S, O))
        print("j = %s, m = %s: sigma.L Omega = %d Omega:" % (j, m, ev), ok)
O = Omega(QQ(1) / 2, 1, QQ(1) / 2)
print("Omega_{1/2,1,1/2} =", [o.simplify_full() for o in O], "; |Omega|^2 =", sum((o * o.conjugate()).simplify_full() for o in O).simplify_full())
