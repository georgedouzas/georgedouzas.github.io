# Exercise exr-compose-rotations-noncommute: rotations about the Z- and X-axes do not commute.
a, b = var("a b")
Rz = matrix(SR, [[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])
Rx = matrix(SR, [[1, 0, 0], [0, cos(b), -sin(b)], [0, sin(b), cos(b)]])
D = (Rz * Rx - Rx * Rz).simplify_full()
print("Rz Rx - Rx Rz ="); print(D)
print("[PASS] the (1,3) entry of Rz Rx is sin a sin b while that of Rx Rz is 0:", bool((Rz * Rx)[0, 2] == sin(a) * sin(b)) and bool((Rx * Rz)[0, 2] == 0))
