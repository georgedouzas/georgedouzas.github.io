# Exercise exr-count-maps-4-3: maps, injections and surjections from 4 elements to 3.
from itertools import product
maps = list(product("abc", repeat=4))
inj = sum(1 for v in maps if len(set(v)) == 4); sur = sum(1 for v in maps if set(v) == {"a", "b", "c"})
print("maps:", len(maps), "; injective:", inj, "; surjective:", sur)
print("[PASS] 81 maps, 0 injective, 36 surjective:", (len(maps), inj, sur) == (81, 0, 36))
