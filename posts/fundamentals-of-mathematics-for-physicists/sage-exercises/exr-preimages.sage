# Exercise exr-preimages: continuity through preimages, checked on a grid. The step function has a preimage of an
# open interval that is a half-closed set; the cosine has open preimages of open intervals.
step = lambda x: 0 if x < 0 else 1
pts = [k / 10 for k in range(-20, 21)]
pre_step = [x for x in pts if 1/2 < step(x) < 2]
print("step: preimage of (1/2, 2) on the grid starts at", min(pre_step), "which is included (the set [0, oo) is not open)")
pre_cos = [x for x in pts if 1/2 < cos(x) < 2]
print("cos: preimage of (1/2, 2) on the grid is", min(pre_cos), "..", max(pre_cos), "(the open interval (-pi/3, pi/3) contains it)")
print("endpoints excluded:", bool(cos(pi/3) == 1/2), "-> cos(pi/3) = 1/2 is not in the open preimage")
