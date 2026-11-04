# Exercise exr-compose-associative: associativity of composition on three random permutations of {1..5}.
G = SymmetricGroup(5)
f, g, h = [G.random_element() for _ in range(3)]
print("f, g, h =", f, g, h)
print("[PASS] (h g) f = h (g f):", (f * g) * h == f * (g * h))
