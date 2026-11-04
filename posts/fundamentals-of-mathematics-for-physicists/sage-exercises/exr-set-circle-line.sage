# Exercise exr-set-circle-line: the circle cut by a line, the sphere cut by a plane.
X, Y, Z = var("X Y Z")
print("circle and the line Y = 1/2:", solve([X^2 + Y^2 == 1, Y == 1/2], [X, Y]))
print("sphere and the plane Z = 1/2:", (X^2 + Y^2 + Z^2 - 1).subs(Z=1/2), "= 0, a circle of radius", sqrt(3/4))
print("[PASS] two points on the circle, a circle of radius sqrt(3)/2 on the sphere:", len(solve([X^2 + Y^2 == 1, Y == 1/2], [X, Y])) == 2)
