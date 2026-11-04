# Exercise exr-sr-phase-invariant: the phase k.x of a plane wave is invariant under a boost applied to both k and x, and
# k stays null.
var('t x y z w th v', domain='real')
g = 1/sqrt(1 - v^2)
L = matrix(SR, [[g, -g*v, 0, 0], [-g*v, g, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
eta = diagonal_matrix([1, -1, -1, -1])
X = vector([t, x, y, z]); k = w*vector([1, cos(th), sin(th), 0])
print("k'.x' - k.x =", ((L*k)*eta*(L*X) - k*eta*X).simplify_full())
print("k'.k' =", ((L*k)*eta*(L*k)).simplify_full())
