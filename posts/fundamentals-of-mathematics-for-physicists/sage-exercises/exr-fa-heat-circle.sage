# Exercise exr-fa-heat-circle: the theta kernel on the circle, (1/2pi) sum e^{-n^2 t} e^{inx}, equals the periodised
# heat kernel sum_k e^{-(x - 2 pi k)^2/4t}/sqrt(4 pi t) (Poisson summation), checked numerically at t = 0.1 and t = 1.
for tv in [0.1, 1]:
    for xv in [0, 1, 3]:
        fourier = (sum(exp(-n ** 2 * tv) * cos(n * xv) for n in range(-40, 41)) / (2 * pi)).n(digits=12)
        images = sum(exp(-(xv - 2 * pi * k) ** 2 / (4 * tv)) / sqrt(4 * pi * tv) for k in range(-10, 11)).n(digits=12)
        f3 = (sum(exp(-n ** 2 * tv) * cos(n * xv) for n in range(-3, 4)) / (2 * pi)).n(digits=12)
        i1 = sum(exp(-(xv - 2 * pi * k) ** 2 / (4 * tv)) / sqrt(4 * pi * tv) for k in range(-1, 2)).n(digits=12)
        print("t =", tv, "x =", xv, ": Fourier =", fourier, "; images =", images, "; agree:", bool(abs(fourier - images) < 1e-10), "; error of 7 Fourier terms", abs(f3 - fourier).n(digits=2), "; of 3 images", abs(i1 - images).n(digits=2))
