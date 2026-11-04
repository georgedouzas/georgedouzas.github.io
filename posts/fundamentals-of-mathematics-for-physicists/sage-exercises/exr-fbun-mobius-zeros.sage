# Exercise exr-fbun-mobius-zeros: sections of the Moebius band are functions with s(phi + 2 pi) = -s(phi); the
# sections cos((2k+1) phi / 2) have 2k+1 zeros on [0, 2 pi), and a perturbed section still has an odd number.
ph = var("ph", domain="real")
for k in range(3):
    s = cos((2 * k + 1) * ph / 2)
    roots = sorted(set(n((2 * j + 1) * pi / (2 * k + 1)) for j in range(-5, 10) if 0 <= (2 * j + 1) * pi / (2 * k + 1) < 2 * pi))
    print(f"s = cos({2*k+1} phi/2): s(phi + 2 pi) = -s(phi):", bool((s.subs(ph=ph + 2 * pi) + s).simplify_full() == 0),
          "; zeros in [0, 2 pi):", len(roots))
s = cos(ph / 2) + 3 / 5 * sin(3 * ph / 2) + 1 / 4 * cos(5 * ph / 2)
print("perturbed section antiperiodic:", bool((s.subs(ph=ph + 2 * pi) + s).simplify_full().trig_reduce() == 0))
f = fast_callable(s, vars=[ph], domain=RDF)
N = 20000; grid = [2 * float(pi) * j / N for j in range(N + 1)]
changes = sum(1 for j in range(N) if f(grid[j]) * f(grid[j + 1]) < 0)
print("sign changes of the perturbed section on [0, 2 pi]:", changes, "(odd:", changes % 2 == 1, ")")
