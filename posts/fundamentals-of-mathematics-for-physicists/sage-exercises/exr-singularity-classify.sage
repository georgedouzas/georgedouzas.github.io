# Exercise exr-singularity-classify: the singularity at 0 of sin z/z, (1 - cos z)/z^4 and z e^{1/z} from their
# series: removable, a pole of order 2, essential (the last by substituting w = 1/z into the series of e^w).
z, w = var("z w")
for f in (sin(z)/z, (1 - cos(z))/z^4):
    print("f =", f, ":  ", f.series(z == 0, 3))
ess = (z*taylor(exp(w), w, 0, 6).subs(w=1/z)).expand()
print("f = z e^{1/z}: ", ess, "+ ... (the terms z^{1-n}/n! never stop)")
print("sin z/z: no negative powers, removable with value", limit(sin(z)/z, z=0))
print("(1 - cos z)/z^4: lowest power z^-2, pole of order 2; residue", ((1 - cos(z))/z^4).residue(z == 0))
print("z e^{1/z}: essential; coefficient of 1/z is", ess.coefficient(z, -1))
