# Exercise exr-spd-dynkin-ratio: tr(J_3^2) in the spin-J representation divided by its value 1/2 in the fundamental.
for twoJ in range(1, 7):
    J = twoJ / 2
    tr = sum((J - k) ^ 2 for k in range(twoJ + 1))
    print("J = %s: tr J_3^2 = %s, ratio = %s, (2/3) J (J+1)(2J+1) = %s" % (J, tr, tr / (1/2), 2/3 * J * (J + 1) * (2 * J + 1)))
