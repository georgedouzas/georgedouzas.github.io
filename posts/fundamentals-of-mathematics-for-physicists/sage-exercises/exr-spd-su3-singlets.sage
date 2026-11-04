# Exercise exr-spd-su3-singlets: the Dirac weights of Spin(6) invariant under SU(3) (h_1 = h_2 = h_3), with chiralities.
import itertools
for h in itertools.product((1/2, -1/2), repeat=3):
    chi = (-1) ^ 3 * prod(2 * t for t in h)
    print(h, "chirality %+d" % chi, "<- SU(3) singlet" if h[0] == h[1] == h[2] else "")
