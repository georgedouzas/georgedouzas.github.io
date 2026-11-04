# Exercise exr-ren-pi-wick: Wick ordering :A:_c = e^(-c d^2/2) A on polynomials, and E_c theta :phi^6:_c = phi^6.
var('phi c z')
def heat(A, s):            # e^(s d^2/2) A as a finite sum
    out, term, k = 0, A, 0
    while term != 0:
        out += (s / 2)^k / factorial(k) * term
        term = diff(term, phi, 2); k += 1
    return out.expand()
print(":phi^4:_c =", heat(phi^4, -c))
print(":phi^6:_c =", heat(phi^6, -c))
W6 = heat(phi^6, -c)
conv = integrate(W6.subs(phi=phi + z) * exp(-z^2 / (2 * c)) / sqrt(2 * pi * c), z, -oo, oo, algorithm='maxima') if False else heat(W6, c)
print("E_c theta :phi^6:_c =", conv)
assume(c > 0)
direct = integrate(W6.subs(phi=phi + z) * exp(-z^2 / (2 * c)), z, -oo, oo) / sqrt(2 * pi * c)
print("by integration against the Gaussian:", direct.simplify_full().expand())
