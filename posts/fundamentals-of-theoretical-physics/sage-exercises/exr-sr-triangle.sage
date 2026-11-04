# Exercise exr-sr-triangle (Blennow-Ohlsson, Problem 1.16): a 3-4-5 right triangle seen from frames moving parallel to a
# leg and to the hypotenuse; sides and area (shoelace formula) with only the x-components contracted.
var('g', domain='positive')
def seen(P):
    Q = [vector([p[0]/g, p[1]]) for p in P]
    sides = [sqrt((Q[i] - Q[(i + 1) % 3])*(Q[i] - Q[(i + 1) % 3])).simplify_full() for i in range(3)]
    area = abs(sum(Q[i][0]*Q[(i + 1) % 3][1] - Q[(i + 1) % 3][0]*Q[i][1] for i in range(3))/2).simplify_full()
    return sides, area
# (a) leg a = 3 on the x-axis, right angle at the origin
print("(a)", seen([vector([0, 0]), vector([3, 0]), vector([0, 4])]))
# (b) hypotenuse on the x-axis from (0,0) to (5,0), right-angle vertex at (9/5, 12/5)
print("(b)", seen([vector([0, 0]), vector([5, 0]), vector([9/5, 12/5])]))
