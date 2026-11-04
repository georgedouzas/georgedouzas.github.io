# Exercise exr-sr-null-rotation-exp: N = K1 - J2 is nilpotent of order three and exp(a N) is the null rotation, the image
# of the unipotent matrix [[1, a], [0, 1]] under the covering map.
var('a', domain='real')
K1 = matrix(SR, 4, 4); K1[0, 1] = K1[1, 0] = 1
J2 = matrix(SR, 4, 4); J2[1, 3] = 1; J2[3, 1] = -1
N = K1 - J2
print("N^2 ="); print(N^2); print("N^3 = 0:", N^3 == 0)
E = identity_matrix(SR, 4) + a*N + a^2/2*N^2
s = [identity_matrix(SR, 2), matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
A = matrix(SR, [[1, a], [0, 1]])
L = matrix(SR, 4, 4, lambda m, n: ((s[m]*A*s[n]*A.conjugate_transpose()).trace()/2).simplify_full())
print("Lambda([[1,a],[0,1]]) ="); print(L)
print("equals exp(a N):", (L - E).simplify_full() == 0)
