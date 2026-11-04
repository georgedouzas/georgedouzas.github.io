# Exercise exr-lgt-st-z2-table: heat-bath probabilities of the 3D Z2 gauge theory for every staple sum.
for beta in (0.5, 0.76):
    print('beta = %.2f:' % beta, '  '.join('s = %+d: %.4f' % (s, exp(beta * s) / (2 * cosh(beta * s))) for s in range(-4, 5, 2)))
print('Boudreau-Swanson weights exp(beta_BS s / 2) with beta_BS = 2 beta equal exp(beta s): the same table')
