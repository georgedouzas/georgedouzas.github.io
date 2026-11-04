# Exercise exr-emx-mm-sphere-check: dielectric sphere in a uniform field.
r, th, E0, a, eps = var('r theta E0 a epsilon', domain='positive')
pin = -3 * E0 / (eps + 2) * r * cos(th)
pout = -E0 * r * cos(th) + (eps - 1) / (eps + 2) * E0 * a^3 * cos(th) / r^2
def lap(f):
    return (diff(r^2 * diff(f, r), r) / r^2 + diff(sin(th) * diff(f, th), th) / (r^2 * sin(th))).simplify_full()
print('harmonic:', lap(pin), lap(pout))
print('phi continuous:', bool((pin - pout).subs(r=a).simplify_full() == 0))
print('eps d_r phi(in) = d_r phi(out):', bool((eps * diff(pin, r) - diff(pout, r)).subs(r=a).simplify_full() == 0))
