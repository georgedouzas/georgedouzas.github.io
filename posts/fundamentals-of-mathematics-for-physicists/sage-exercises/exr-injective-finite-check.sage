# Exercise exr-injective-finite-check: injectivity of listed maps and the count of injective maps 3 -> 4.
from itertools import product
maps = {"(a,b,c)": ("a", "b", "c"), "(a,a,b)": ("a", "a", "b"), "(d,c,b)": ("d", "c", "b")}
inj = lambda vals: len(set(vals)) == len(vals)
for k, v in maps.items(): print(k, "injective:", inj(v))
count = sum(1 for vals in product("abcd", repeat=3) if inj(vals))
print("[PASS] injective maps from 3 to 4 elements: 4*3*2 = 24:", count == 24, count)
