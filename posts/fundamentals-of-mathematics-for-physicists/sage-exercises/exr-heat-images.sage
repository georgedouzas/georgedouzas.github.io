# Exercise exr-heat-images (Foulaadvand, Sec. 5.2 and Problem 5.3): the heat kernel on [-L/2, L/2] with u = 0 at the
# ends, from the alternating sum of images sum (-1)^n K(t, x + nL), against the sine series obtained by separation of
# variables, sum (2/L) sin(k_j (x + L/2)) sin(k_j L/2) e^(-k_j^2 t), k_j = j pi/L (kappa = 1, L = 1), in floating point.
from math import exp, sqrt, pi, sin
L = 1.0
K = lambda t, x: exp(-x^2/(4*t))/sqrt(4*pi*t)
images = lambda t, x, M: sum((-1)^n*K(t, x + n*L) for n in range(-M, M + 1))
modes = lambda t, x, J: sum(2/L*sin(j*pi*(x + L/2)/L)*sin(j*pi/2)*exp(-(j*pi/L)^2*t) for j in range(1, J + 1))
print("    t     x   images |n|<=1   images |n|<=6   modes j<=5      modes j<=200")
for t in [0.001, 0.01, 0.1, 0.5]:
    for x0 in [0.0, 0.3, 0.5]:
        print("%6.3f  %4.1f  %13.9f  %13.9f  %13.9f  %13.9f" % (t, x0, images(t, x0, 1), images(t, x0, 6), modes(t, x0, 5), modes(t, x0, 200)))
