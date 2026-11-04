# Exercise exr-clm-pa-ladder: the ladder functions L_pm = L_x pm i L_y and their Poisson brackets.
x, y, z, px, py, pz = var("x y z px py pz", domain="real")
qs, ps = [x, y, z], [px, py, pz]
def PB(f, g): return sum(diff(f, qs[i]) * diff(g, ps[i]) - diff(f, ps[i]) * diff(g, qs[i]) for i in range(3))
Lx, Ly, Lz = y * pz - z * py, z * px - x * pz, x * py - y * px
Lp, Lm = Lx + I * Ly, Lx - I * Ly
print("{L_z, L_+} + i L_+ =", (PB(Lz, Lp) + I * Lp).expand())
print("{L_z, L_-} - i L_- =", (PB(Lz, Lm) - I * Lm).expand())
print("{L_+, L_-} + 2 i L_z =", (PB(Lp, Lm) + 2 * I * Lz).expand())
