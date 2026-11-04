# Exercise exr-cff-spinor-weights: the Weyl spinor of SO(6) on the flag manifold SU(3)/(U(1) x U(1)) has the weights
# (1/2)(+-b1 +-b2 +-b3) with an even number of minus signs, b_i the holomorphic tangent weights in the basis of
# Chatzistavrakidis-Zoupanos (2009), eq. (4.31); matched against the decomposition (4.30) of the 248 of E_8.
import itertools
from collections import Counter
b = [vector(QQ, (1, 1)), vector(QQ, (-1, 1)), vector(QQ, (0, -2))]
print("b1 + b2 + b3 =", sum(b))
spin = [tuple(sum(s_ * v for s_, v in zip(signs, b)) / 2) for signs in itertools.product((1, -1), repeat=3) if prod(signs) == 1]
print("Weyl spinor weights:", spin)
e8 = [((0, 0), "1"), ((0, 0), "1"), ((1, 3), "1"), ((-1, 3), "1"), ((2, 0), "1"), ((-2, 0), "1"), ((-1, -3), "1"), ((1, -3), "1"), ((0, 0), "78"),
      ((1, 1), "27"), ((-1, 1), "27"), ((0, -2), "27"), ((-1, -1), "27bar"), ((1, -1), "27bar"), ((0, 2), "27bar")]
kept = Counter(rep for w, rep in e8 if w in spin)
print("surviving fermions by E_6 representation:", dict(kept))
print("dimension check of the decomposition:", sum({"1": 1, "78": 78, "27": 27, "27bar": 27}[r] for _, r in e8))
