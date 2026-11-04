# Exercise exr-continuous-floor: the floor function jumps at the integers.
f = lambda x: floor(x)
jumps = [k for k in range(-2, 3) if f(k) != f(k - 1e-9)]
print("jump points among -2..2:", jumps)
print("preimage of (k - 1/2, k + 1/2) for k = 1 is [1, 2): endpoints", f(1), f(1.999), "and", f(0.999), "outside")
print("[PASS] discontinuous exactly at the integers (sampled), continuous between:", jumps == [-2, -1, 0, 1, 2] and f(1.2) == f(1.7))
