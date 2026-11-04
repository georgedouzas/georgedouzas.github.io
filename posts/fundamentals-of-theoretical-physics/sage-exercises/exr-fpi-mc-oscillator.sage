# Exercise exr-fpi-mc-oscillator: Metropolis sampling of the Euclidean oscillator on a periodic time lattice,
# S = sum_n [ (q_{n+1} - q_n)^2/(2a) + a omega^2 q_n^2/2 ], N = 20 sites, a = 1/2, omega = 1 (beta = N a = 10).
# The estimate of <q^2>, with its error from blocks of sweeps, is compared with the exact lattice value
# a coth(N eps/2)/(2 sinh eps), 2 sinh(eps/2) = omega a, and with the continuum value 1/(2 omega) at beta = oo.
# The loop is plain Python (executed without the Sage preparser, for speed).
exec(r'''
import random, math
rng = random.Random(20261019)
N, a, w, Delta = 20, 0.5, 1.0, 1.0
q = [0.0] * N
def dS(n, new):                       # change of the action when q_n -> new (only the neighbours of n enter)
    old, l, r = q[n], q[(n - 1) % N], q[(n + 1) % N]
    kin = ((r - new)**2 + (new - l)**2 - (r - old)**2 - (old - l)**2) / (2 * a)
    return kin + a * w**2 * (new**2 - old**2) / 2
accepted, therm, sweeps, block = 0, 1000, 100000, 2000
q2, c1, c2, blocks = 0.0, 0.0, 0.0, []
for sweep in range(therm + sweeps):
    for n in range(N):
        new = q[n] + rng.uniform(-Delta, Delta)
        d = dS(n, new)
        if d <= 0 or rng.random() < math.exp(-d):
            q[n] = new
            if sweep >= therm: accepted += 1
    if sweep >= therm:
        m2 = sum(x * x for x in q) / N
        q2 += m2; blocks.append(m2)
        c1 += sum(q[n] * q[(n + 1) % N] for n in range(N)) / N
        c2 += sum(q[n] * q[(n + 2) % N] for n in range(N)) / N
means = [sum(blocks[i:i + block]) / block for i in range(0, sweeps, block)]
mean = sum(means) / len(means)
err = math.sqrt(sum((m - mean)**2 for m in means) / (len(means) * (len(means) - 1)))
eps = 2 * math.asinh(w * a / 2)
exact = lambda n: a * math.cosh(eps * (N / 2 - n)) / (2 * math.sinh(eps) * math.sinh(eps * N / 2))
print(f"acceptance rate {accepted / (sweeps * N):.3f} with Delta = {Delta}")
print(f"Monte Carlo <q^2> = {mean:.4f} +- {err:.4f} ({len(means)} blocks of {block} sweeps)")
print(f"exact lattice <q^2> = {exact(0):.4f}; continuum 1/(2 omega) = {1 / (2 * w):.4f}")
print(f"<q_n q_(n+1)> = {c1 / sweeps:.4f} (exact {exact(1):.4f}); <q_n q_(n+2)> = {c2 / sweeps:.4f} (exact {exact(2):.4f})")
print(f"effective mass log(<q^2>/<q_n q_(n+1)>)/a = {math.log(q2 / c1) / a:.3f}; lattice E = eps/a = {eps / a:.4f}; omega = {w}")
''')
