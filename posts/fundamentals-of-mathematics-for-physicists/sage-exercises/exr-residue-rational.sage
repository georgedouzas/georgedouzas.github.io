# Exercise exr-residue-rational: residues of (z^2 + 1)/(z (z - 1)^2) at 0 (simple pole) and 1 (double pole), by the
# limit formulas and by Sage's residue; the residues including infinity sum to zero.
z = var("z")
f = (z^2 + 1)/(z*(z - 1)^2)
r0 = limit(z*f, z=0)
r1 = limit(diff((z - 1)^2*f, z), z=1)
print("by the formulas: Res_0 =", r0, " Res_1 =", r1)
print("by Sage:         Res_0 =", f.residue(z == 0), " Res_1 =", f.residue(z == 1))
rinf = -(f.subs(z=1/z)/z^2).residue(z == 0)      # residue at infinity
print("Res_infinity =", rinf, "  sum of all residues:", r0 + r1 + rinf)
