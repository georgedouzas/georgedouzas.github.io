# Exercise exr-derivative-rules: derivative of f(z) = (z - 1)/(z^2 + 1) by the quotient rule and its zeros.
z = var("z")
f = (z - 1)/(z^2 + 1)
fp = diff(f, z).factor()
print("f'(z) =", fp)
print("numerator -z^2 + 2z + 1 (quotient rule):", bool((fp*(z^2 + 1)^2).simplify_full() == -z^2 + 2*z + 1))
print("zeros of f':", solve(fp == 0, z))
