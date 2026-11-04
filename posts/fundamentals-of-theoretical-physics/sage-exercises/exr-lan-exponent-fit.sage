# Exercise exr-lan-exponent-fit: estimate the exponents beta (m ~ (-t)^beta) and delta (m ~ h^(1/delta) at T_c) from the
# mean-field Ising equation m = tanh((m + h) T_c/T), by log-log slopes of exact numerical solutions.
def msol(t, h):
    return find_root(lambda x: x - tanh((x + h) / t), 1e-12, 1.0)
ts = [1 - 10 ** (-k) for k in [3, 4, 5, 6]]
ms = [msol(t, 0) for t in ts]
for k in range(len(ts) - 1):
    print("beta estimate from 1 - T/T_c = %.0e, %.0e: %.5f" % (1 - ts[k], 1 - ts[k + 1], (log(ms[k + 1]) - log(ms[k])) / (log(1 - ts[k + 1]) - log(1 - ts[k]))))
hs = [10 ** (-k) for k in [4, 5, 6, 7]]
mh = [msol(1, h) for h in hs]
for k in range(len(hs) - 1):
    print("1/delta estimate from h = %.0e, %.0e: %.5f" % (hs[k], hs[k + 1], (log(mh[k + 1]) - log(mh[k])) / (log(hs[k + 1]) - log(hs[k]))))
print("at T_c, m = (3 h)^(1/3) to leading order: h = 1e-7 gives m = %.6e vs %.6e" % (mh[-1], (3e-7) ** (1 / 3)))
