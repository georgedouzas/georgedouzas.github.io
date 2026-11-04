# Exercise exr-emm-light-sail (Macchi et al., Problem 8.10): a 1 g sail reflecting a 100 GW laser, non-relativistic estimate.
P, m, c = 1e11, 1e-3, 2.998e8
F = 2 * P / c; a = F / m; t = 0.2 * c / a
print('F = %.0f N, a = %.2e m/s^2 = %.0f g, t = %.0f s, distance = %.2e km' % (F, a, a / 9.81, t, a * t^2 / 2 / 1e3))
