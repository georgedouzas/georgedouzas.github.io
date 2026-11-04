# Exercise exr-pfn-equipartition: <c |q|^s> = T/s for the Boltzmann weight e^{-c |q|^s / T}, s = 1, 2, 3, 4, 6;
# s = 2 is equipartition.
q, c, beta = var("q c beta"); assume(q > 0, c > 0, beta > 0)
for s in [1, 2, 3, 4, 6]:
    Z = 2 * integrate(exp(-beta * c * q ** s), q, 0, oo)
    E = 2 * integrate(c * q ** s * exp(-beta * c * q ** s), q, 0, oo) / Z
    print("s = %d: <c |q|^s> =" % s, E.simplify_full(), "; equals T/s:", bool((E - 1 / (s * beta)).simplify_full() == 0))
