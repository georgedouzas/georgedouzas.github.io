# Exercise exr-lg-comp-hamilton: Hamilton's theorem. For a spherical triangle A, B, C with det(A, B, C) > 0 and interior
# angles alpha, beta, gamma, R(2 alpha, A) R(2 beta, B) R(2 gamma, C) = 1; each factor is a product of two half-turns
# about the normals of the sides, H(p) H(q) with p, q, r normal to CA, AB, BC.
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
def rot(t, n):
    n = vector(SR, n) / vector(SR, n).norm(); N = sum(n[a] * L[a] for a in range(3))
    return (identity_matrix(SR, 3) + sin(t) * N + (1 - cos(t)) * N^2).apply_map(lambda e: e.simplify_full())
def angle(P, Q, R):  # interior angle at P between the arcs PQ and PR
    u = Q - (P * Q) * P; v = R - (P * R) * P
    return arccos((u * v / (u.norm() * v.norm())).simplify_full())
def check(A, B, C):
    A, B, C = [vector(SR, X) / vector(SR, X).norm() for X in (A, B, C)]
    al, be, ga = angle(A, B, C), angle(B, C, A), angle(C, A, B)
    M = rot(2 * al, A) * rot(2 * be, B) * rot(2 * ga, C)
    H = lambda u: rot(pi, u)
    p, q, r = C.cross_product(A), A.cross_product(B), B.cross_product(C)
    halves = [bool((H(p) * H(q) - rot(2 * al, A)).apply_map(lambda e: e.simplify_full()) == 0),
              bool((H(q) * H(r) - rot(2 * be, B)).apply_map(lambda e: e.simplify_full()) == 0),
              bool((H(r) * H(p) - rot(2 * ga, C)).apply_map(lambda e: e.simplify_full()) == 0)]
    print("det(A,B,C) =", matrix([A, B, C]).det().simplify_full(), "; angles", [al, be, ga],
          "; product = 1:", bool((M - 1).apply_map(lambda e: e.simplify_full()) == 0), "; factors are H(p)H(q), H(q)H(r), H(r)H(p):", halves)
check([1, 0, 0], [0, 1, 0], [0, 0, 1])      # the octant: three half-turns
check([1, 0, 0], [0, 1, 0], [1, 1, 1])      # an isosceles triangle, angles pi/4, pi/4, 2 pi/3
# a generic triangle, numerically
A, B, C = vector(RDF, [1, 0, 0]), vector(RDF, [0.3, 1, 0.2]), vector(RDF, [0.1, 0.4, 1.3])
A, B, C = A / A.norm(), B / B.norm(), C / C.norm()
def rotN(t, n):
    N = sum(n[a] * L[a].change_ring(RDF) for a in range(3)); return identity_matrix(RDF, 3) + sin(t) * N + (1 - cos(t)) * N^2
def angN(P, Q, R):
    u = Q - (P * Q) * P; v = R - (P * R) * P; return arccos(u * v / (u.norm() * v.norm()))
al, be, ga = angN(A, B, C), angN(B, C, A), angN(C, A, B)
print("generic triangle: angle sum - pi =", round(al + be + ga - pi.n(), 6), "(the area); |R(2a,A)R(2b,B)R(2c,C) - 1| =",
      round((rotN(2 * al, A) * rotN(2 * be, B) * rotN(2 * ga, C) - identity_matrix(RDF, 3)).norm(), 12),
      "; reversed order:", round((rotN(2 * ga, C) * rotN(2 * be, B) * rotN(2 * al, A) - identity_matrix(RDF, 3)).norm(), 6))
