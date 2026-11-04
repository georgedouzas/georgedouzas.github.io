# Exercise exr-dq-seesaw-sage: eigenvalues of [[0, mD], [mD, M]], the light mass to third order in mD/M, and its value for mD = 100 GeV,
# M = 10^14 GeV.
mD, M, x = var("mD M x", domain="positive")
ev = matrix(SR, [[0, mD], [mD, M]]).eigenvalues()
print("eigenvalues:", ev)
light = (sqrt(M^2 + 4 * mD^2) - M) / 2
print("light mass:", taylor(light.subs(M=mD / x), x, 0, 3).subs(x=mD / M))
val = light.subs(mD=100, M=10^14)
print("light mass for mD = 100 GeV, M = 1e14 GeV: %.4e GeV = %.4f eV (mD^2/M = %.4e GeV)" % (RealField(200)(val), RealField(200)(val) * 1e9, 100^2 / 10^14))
