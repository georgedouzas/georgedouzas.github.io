# Exercise exr-quotient-z12-classes: classes of Z/12Z containing multiples of 4, and the map Z_12 -> Z_4.
R12 = Zmod(12)
classes = sorted(set(R12(4 * k).lift() for k in range(12)))
print("classes of multiples of 4:", classes)
fibres = {r: [k for k in range(12) if k % 4 == r] for r in range(4)}
print("preimages of k -> k mod 4:", fibres)
print("[PASS] classes [0], [4], [8] and three-element preimages:", classes == [0, 4, 8] and all(len(f) == 3 for f in fibres.values()))
