# Exercise exr-compose-permutations: composition and inverses of bijections of {1, 2, 3} with Sage's SymmetricGroup.
# Sage multiplies permutations left to right: (s*t)(i) = t(s(i)).
S3 = SymmetricGroup(3)
s = S3([(1, 2)])        # swaps 1 and 2
t = S3([(1, 2, 3)])     # 1 -> 2 -> 3 -> 1
print("s =", s, "; t =", t)
print("s then t:", s * t, "; t then s:", t * s, "; equal:", s * t == t * s)
print("inverse of t:", t.inverse(), "; t * t^-1 is the identity:", (t * t.inverse()).is_one())
print("number of bijections of {1,2,3}:", S3.order())
print("every element composed with its inverse gives the identity:", all((g * g.inverse()).is_one() for g in S3))
