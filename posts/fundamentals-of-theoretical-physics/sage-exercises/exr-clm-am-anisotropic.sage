# Exercise exr-clm-am-anisotropic: L_z is not conserved for the anisotropic oscillator H = (px^2 + py^2)/2 + (x^2 + 4y^2)/2.
x, y, px, py, t = var("x y px py t", domain="real")
qs, ps = [x, y], [px, py]
def PB(f, g): return sum(diff(f, qs[i]) * diff(g, ps[i]) - diff(f, ps[i]) * diff(g, qs[i]) for i in range(2))
H = (px**2 + py**2) / 2 + (x**2 + 4 * y**2) / 2
Lz = x * py - y * px
print("{L_z, H} =", PB(Lz, H).expand())
sol = {x: cos(t), y: sin(2 * t) / 2, px: -sin(t), py: cos(2 * t)}
print("solution satisfies Hamilton's equations:", all(bool((diff(sol[a], t) - diff(H, b).subs(sol)).simplify_full() == 0) for a, b in [(x, px), (y, py)]) and all(bool((diff(sol[b], t) + diff(H, a).subs(sol)).simplify_full() == 0) for a, b in [(x, px), (y, py)]))
Lt = Lz.subs(sol)
print("L_z(t) =", Lt.simplify_full(), "; at t = 0:", Lt.subs(t=0), "; at t = pi:", Lt.subs(t=pi))
vals = [Lt.subs(t=k * 2 * pi / 400).n() for k in range(401)]
print("min, max of L_z(t) over a period: %.4f %.4f" % (min(vals), max(vals)))
