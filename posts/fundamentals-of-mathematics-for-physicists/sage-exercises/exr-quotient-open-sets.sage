# Exercise exr-quotient-open-sets: the preimage of the image of an open interval under R -> R/2piZ is the union of
# its translates, a 2 pi periodic open set.
U = (0.5, 1.2)
in_preimage = lambda t: any(U[0] < t - 2 * k * float(pi) < U[1] for k in range(-3, 4))
samples = [0.7, 0.7 + 2 * float(pi), 0.7 - 4 * float(pi), 2.0, 2.0 + 2 * float(pi)]
print([(round(t, 2), in_preimage(t)) for t in samples])
print("[PASS] membership is 2 pi periodic (translates of the interval are in, others out):", [in_preimage(t) for t in samples] == [True, True, True, False, False])
