# Exercise exr-tsm-thermal-scalar: a free massless real scalar in 3 + 1 dimensions at temperature T. Each momentum mode
# is an oscillator with omega = |k|, whose free energy omega/2 + T log(1 - e^{-omega/T}) follows from the Matsubara sum;
# the thermal part integrates to f = -pi^2 T^4/90, and two such fields (photon polarisations) give -pi^2 T^4/45.
k, T, x, n = var("k T x n"); assume(k > 0, T > 0, x > 0, n > 0)
# int x^2 log(1 - e^{-x}) dx = -(1/3) int x^3/(e^x - 1) dx = -(1/3) sum_n 6/n^4
I3 = sum(integrate(x ** 3 * exp(-n * x), x, 0, oo), n, 1, oo)
f = 4 * pi / (2 * pi) ** 3 * T ** 4 * (-I3 / 3)
print("f = (T^4/(2 pi^2)) int x^2 log(1 - e^{-x}) dx =", f, "; -pi^2 T^4/90:", bool(f == -pi ** 2 * T ** 4 / 90))
print("numerical check of int x^2 log(1 - e^{-x}) dx:", numerical_integral(x ** 2 * log(1 - exp(-x)), 0, 80)[0], "vs", (-pi ** 4 / 45).n())
print("s = -df/dT =", -diff(f, T), "; u = f + T s =", (f - T * diff(f, T)), "; p = -f, p = u/3:", bool(-f == (f - T * diff(f, T)) / 3))
# the tower: masses 2 pi n T of the three-dimensional fields, and the circumference 1/T
for Tv in [1, 10]:
    print("T = %s: lowest bosonic masses 2 pi |n| T =" % Tv, [(2 * pi * abs(j) * Tv).n(digits=4) for j in range(0, 3)], "; lowest fermionic pi T =", (pi * Tv).n(digits=4))
