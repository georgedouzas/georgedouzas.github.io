# Exercise exr-emx-sc-degree: degree of polarisation of Rayleigh-scattered light.
th = var('th')
Pd = (1 - cos(th)^2) / (1 + cos(th)^2)
for d in [30, 60, 90]:
    print(d, 'deg:', Pd.subs({th: d * pi / 180}), '=', n(Pd.subs({th: d * pi / 180}), digits=3))
