# Exercise exr-emx-lin-coax-resistor: radial resistance R = ln(b/a)/(2 pi sigma L) (SI).
R = lambda a, b, s, L: log(b / a) / (2 * pi * s * L)
print('sea water, 1 m:', n(R(0.01, 0.1, 4, 1), digits=3), 'ohm')
print('polyethylene cable, 1 km:', n(R(0.5e-3, 1.75e-3, 1e-15, 1000), digits=3), 'ohm')
