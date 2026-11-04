# Exercise exr-laurent-exp-1-over-z: Laurent series of e^{1/z} and of z^2 e^{1/z} about 0 and their residues, by
# substituting w = 1/z into the Taylor series of e^w (Sage's series/residue at an essential singularity are not
# reliable: exp(1/z).residue(z == 0) returns 0).
z, w = var("z w")
expw = taylor(exp(w), w, 0, 6)
for pre in (1, z^2):
    f = pre*exp(1/z)
    s = (pre*expw.subs(w=1/z)).expand()
    print("f =", f)
    print("   Laurent series (terms to w^6):", s)
    print("   coefficient of 1/z (residue):", s.coefficient(z, -1))
print("warning: Sage's exp(1/z).residue(z == 0) returns", exp(1/z).residue(z == 0), "(wrong; the series above shows 1)")
