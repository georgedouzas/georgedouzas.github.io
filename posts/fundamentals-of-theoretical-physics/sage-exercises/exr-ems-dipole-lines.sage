# Exercise exr-ems-dipole-lines (Pierrus, Question 2.15): equipotentials r^2 = A cos(theta), field lines r = B sin(theta)^2.
th, A, B = var('theta A B', domain='positive')
# tangent of a curve r(theta) in the (r, r dtheta) frame is (dr/dtheta, r)
r_eq = sqrt(A * cos(th)); r_fl = B * sin(th)^2
t_eq = vector([diff(r_eq, th), r_eq]); t_fl = vector([diff(r_fl, th), r_fl])
# at an intersection both r agree; use r for the second component of both
rr = var('rr')
dot = (diff(r_eq, th) / r_eq * rr) * (diff(r_fl, th) / r_fl * rr) + rr * rr
print("inner product of the tangents (divided by r^2):", (dot / rr^2).simplify_full())
p = var('p'); phi = p * cos(th) / (4 * pi * var('r')^2)
r = var('r')
Er = -diff(phi, r); Eth = -diff(phi, th) / r
print("E_r/E_theta =", (Er / Eth).simplify_full(), "= 2 cot(theta)")
