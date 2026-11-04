# Exercise exr-fld-wave-beats: two Klein-Gordon plane waves of nearby wave numbers k1, k2 add up to a carrier
# cos(kbar x - wbar t) times an envelope cos(dk x/2 - dw t/2); the envelope moves at dw/dk, which tends to the group
# velocity k/sqrt(k^2 + m^2) as k2 -> k1, while the carrier moves at the phase velocity.
x, t, m = var("x t m", domain="real"); assume(m > 0)
k1, k2 = var("k1 k2", domain="real"); assume(k1 > 0, k2 > 0)
w = lambda k: sqrt(k**2 + m**2)
W1, W2 = var("W1 W2", domain="real")                   # stand for w(k1), w(k2); the identity holds for any values
s = cos(k1 * x - W1 * t) + cos(k2 * x - W2 * t)
kb, wb, dk, dw = (k1 + k2) / 2, (W1 + W2) / 2, k2 - k1, W2 - W1
prod = 2 * cos(dk * x / 2 - dw * t / 2) * cos(kb * x - wb * t)
print("sum = 2 cos(dk x/2 - dw t/2) cos(kbar x - wbar t):", bool((s - prod).trig_reduce().simplify_full() == 0))
h = var("h"); k = var("k", domain="real"); assume(k > 0)
v_env = ((w(k + h) - w(k)) / h)
print("envelope velocity dw/dk as k2 -> k1:", limit(v_env, h=0), " ; phase velocity:", w(k) / k)
print("at k = 3/4, m = 1: group velocity", (k / w(k)).subs(k=3/4, m=1), ", phase velocity", (w(k) / k).subs(k=3/4, m=1))
