# Exercise exr-orbit-stabiliser: orbits and stabilisers with Sage's permutation groups: S_3 acting on {1, 2, 3},
# the dihedral group D_6 acting on the vertices of a hexagon, and the orbit-stabiliser count |G| = |orbit| |stabiliser|.
S3 = SymmetricGroup(3)
orb = S3.orbit(1); stab = S3.stabilizer(1)
print("S_3 acting on {1,2,3}: orbit of 1 =", orb, "; stabiliser of 1 =", stab.list(), "; |orbit| |stab| =", len(orb) * stab.order(), "= |S_3| =", S3.order())
D6 = DihedralGroup(6)
orb6 = D6.orbit(1); stab6 = D6.stabilizer(1)
print("D_6 (order", D6.order(), ") on the hexagon: orbit of vertex 1 =", orb6, "; stabiliser =", stab6.list(), "; |orbit| |stab| =", len(orb6) * stab6.order())
# the cosets of the stabiliser are in bijection with the orbit. Sage composes permutations left to right, so the coset
# {h*g : h in Stab} (Sage's "right" coset) is the set of elements sending vertex 1 where g does.
cosets = D6.cosets(stab6, side="right")
print("number of left cosets of the stabiliser:", len(cosets), "= size of the orbit:", len(orb6))
print("each coset sends vertex 1 to a single point:", all(len({g(1) for g in C}) == 1 for C in cosets))
