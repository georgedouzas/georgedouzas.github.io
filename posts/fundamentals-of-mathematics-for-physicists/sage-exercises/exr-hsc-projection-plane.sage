# Exercise exr-hsc-projection-plane: the projection of (1, 2, 3) onto the plane x + y + z = 0 from two different
# orthonormal bases of the plane, and from the normal vector; all three agree.
u = vector(QQ, [1, 2, 3])
def project(basis):
    W, mu = matrix(QQ, basis).gram_schmidt()
    return sum((w * u) / (w * w) * w for w in W.rows())
p1 = project([[1, -1, 0], [1, 0, -1]]); p2 = project([[0, 1, -1], [1, -1, 0]])
n = vector(QQ, [1, 1, 1]); p3 = u - (n * u) / (n * n) * n
print("from the first basis:", p1, " from the second:", p2, " u - <n, u> n:", p3, " all equal:", p1 == p2 == p3)
print("the result lies in the plane (sum of coordinates 0):", sum(p1) == 0, " and u - Pu is normal to it:", (u - p1) == (n * u) / (n * n) * n)
