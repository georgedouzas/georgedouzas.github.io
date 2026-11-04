# Exercise exr-homo-kernel-image: kernel and image of k -> 3k on Z_12 and of the sign on S_4; in both cases
# |kernel| x |image| = |G|.
R = Zmod(12)
ker = [k for k in range(12) if R(3 * k) == 0]
img = sorted({int(R(3 * k)) for k in range(12)})
print("k -> 3k on Z_12: kernel", ker, "; image", img, "; |ker| |im| =", len(ker) * len(img))
S4 = SymmetricGroup(4)
kerS = [g for g in S4 if g.sign() == 1]
print("sign on S_4: |kernel| =", len(kerS), "; image", sorted({g.sign() for g in S4}), "; |ker| |im| =", len(kerS) * 2)
print("the kernel of the sign is a normal subgroup:", S4.subgroup(kerS).is_normal(S4))
