# Exercise exr-roots-cube: the three cube roots of unity, their sum, and omega^2 = conj(omega) = 1/omega.
w = exp(2*pi*I/3)
roots = [SR(1), w, w^2]
print("cube roots of unity:", [r.simplify_full() for r in roots])
print("sum is 0:", bool(sum(roots).simplify_full() == 0))
print("omega^2 == conj(omega):", bool((w^2 - conjugate(w)).simplify_full() == 0), "  omega^2 == 1/omega:", bool((w^2 - 1/w).simplify_full() == 0))
z = var("z")
print("roots of z^3 - 1 over QQbar:", [r for r, m in (z^3 - 1).roots(ring=QQbar)])
