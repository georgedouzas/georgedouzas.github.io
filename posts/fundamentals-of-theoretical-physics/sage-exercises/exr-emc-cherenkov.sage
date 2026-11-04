# Exercise exr-emc-cherenkov: Cherenkov threshold and angle for an electron in water.
nn = 1.33; mc2 = 0.51099895     # MeV
g = 1 / sqrt(1 - 1 / nn^2)
print('threshold gamma =', n(g, digits=3), '; kinetic energy =', n((g - 1) * mc2, digits=3), 'MeV')
print('maximal angle arccos(1/n) =', n(arccos(1 / nn) * 180 / pi, digits=3), 'deg')
