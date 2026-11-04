# Exercise exr-fs-cos-power: cos^4 x and sin^3 x as trigonometric polynomials, by trig_reduce and by comparison with
# the hand expansions.
x = var("x")
c4 = (cos(x) ** 4).trig_reduce(); s3 = (sin(x) ** 3).trig_reduce()
print("cos^4 x =", c4)
print("sin^3 x =", s3)
print("cos^4 x = 3/8 + cos(2x)/2 + cos(4x)/8:", bool((c4 - (3 / 8 + cos(2 * x) / 2 + cos(4 * x) / 8)).trig_reduce() == 0))
print("sin^3 x = 3 sin(x)/4 - sin(3x)/4:", bool((s3 - (3 * sin(x) / 4 - sin(3 * x) / 4)).trig_reduce() == 0))
