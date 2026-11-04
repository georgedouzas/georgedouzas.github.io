# Exercise exr-poisson-angular-momentum: Poisson brackets with Sage: the canonical brackets, the so(3) algebra of the
# angular momentum components, and {L_z, H} = 0 for a central potential.
x, y, z, px, py, pz = var("x y z px py pz", domain="real")
qs, ps = [x, y, z], [px, py, pz]
def PB(f, g): return sum(diff(f, qs[i]) * diff(g, ps[i]) - diff(f, ps[i]) * diff(g, qs[i]) for i in range(3))
print("{x, px} =", PB(x, px), "; {x, py} =", PB(x, py), "; {px, py} =", PB(px, py))
Lx, Ly, Lz = y * pz - z * py, z * px - x * pz, x * py - y * px
print("{L_x, L_y} = L_z:", bool((PB(Lx, Ly) - Lz).simplify_full() == 0), "; {L_y, L_z} = L_x:", bool((PB(Ly, Lz) - Lx).simplify_full() == 0), "; {L_z, L_x} = L_y:", bool((PB(Lz, Lx) - Ly).simplify_full() == 0))
V = function("V")(sqrt(x**2 + y**2 + z**2))
H = (px**2 + py**2 + pz**2) / 2 + V
print("{L_z, H} for a central potential:", PB(Lz, H).simplify_full(), " -> L_z is conserved")
print("{L^2, L_z} = 0:", bool(PB(Lx**2 + Ly**2 + Lz**2, Lz).simplify_full() == 0))
