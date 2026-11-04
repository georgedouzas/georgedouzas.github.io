# Exercise exr-lgt-wl-creutz: the Creutz ratio removes the perimeter and constant terms of the Wilson loop.
R, T, sigma, mu, c, alpha, C = var('R T sigma mu c alpha C')
def chi(W):
    return (-log(W(R, T) * W(R - 1, T - 1) / (W(R, T - 1) * W(R - 1, T)))).simplify_full()
W1 = lambda r, t: exp(-sigma * r * t - mu * (2 * r + 2 * t) - c)
W2 = lambda r, t: exp(-t * (sigma * r - alpha / r + C) - mu * 2 * r - c)
print('area + perimeter + constant: chi =', chi(W1))
print('Cornell potential:          chi =', chi(W2).factor(), '=', (sigma + alpha / (R * (R - 1))).simplify_full())
print('Cornell, R = 2, 3, 5 with sigma = 0.05, alpha = 0.3:', [ (sigma + alpha / (r * (r - 1)))(sigma=0.05, alpha=0.3) for r in (2, 3, 5)])
