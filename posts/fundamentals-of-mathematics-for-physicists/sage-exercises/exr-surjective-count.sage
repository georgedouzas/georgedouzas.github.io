# Exercise exr-surjective-count: surjections from 4 to 2 elements and from 3 to 3.
from itertools import product
s42 = sum(1 for v in product("xy", repeat=4) if set(v) == {"x", "y"})
s33 = sum(1 for v in product("abc", repeat=3) if set(v) == {"a", "b", "c"})
print("surjections 4 -> 2:", s42, "; surjections 3 -> 3:", s33)
print("[PASS] 14 and 6:", s42 == 14 and s33 == 6)
