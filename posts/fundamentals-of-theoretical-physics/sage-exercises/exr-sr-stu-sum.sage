# Exercise exr-sr-stu-sum: s + t + u = sum of the squared masses, symbolically and for random two-body kinematics.
var('a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3')
eta = diagonal_matrix([1, -1, -1, -1])
p1, p2, p3 = vector([a0, a1, a2, a3]), vector([b0, b1, b2, b3]), vector([c0, c1, c2, c3]); p4 = p1 + p2 - p3
sq = lambda p: p*eta*p
print("s + t + u - (p1^2 + p2^2 + p3^2 + p4^2) =", (sq(p1 + p2) + sq(p1 - p3) + sq(p1 - p4) - sq(p1) - sq(p2) - sq(p3) - sq(p4)).expand())
