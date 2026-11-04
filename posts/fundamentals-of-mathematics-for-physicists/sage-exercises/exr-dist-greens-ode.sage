# Exercise exr-dist-greens-ode: u = -e^{-|x|}/2 solves u'' - u = delta in the distributional sense, checked against the
# test function e^{-x^2} and through the jump of u'.
x = var("x")
u = -exp(-abs(x)) / 2
phi = exp(-x ** 2)
src = diff(phi, x, 2) - phi
lhs = integrate(-exp(x) / 2 * src, x, -oo, 0) + integrate(-exp(-x) / 2 * src, x, 0, oo)
print("int u (phi'' - phi) dx =", lhs.simplify_full(), "=", lhs.n(digits=15), "; phi(0) =", phi.subs(x=0), "; equal to 12 digits:", bool(abs(lhs.n(digits=15) - 1) < 1e-12))
up = diff(-exp(-x) / 2, x); um = diff(-exp(x) / 2, x)
print("u' jumps at 0 from", um.subs(x=0), "to", up.subs(x=0), ": jump", up.subs(x=0) - um.subs(x=0), "= coefficient of delta")
print("u'' - u = 0 away from 0:", bool((diff(-exp(-x) / 2, x, 2) + exp(-x) / 2).simplify_full() == 0) and bool((diff(-exp(x) / 2, x, 2) + exp(x) / 2).simplify_full() == 0))
