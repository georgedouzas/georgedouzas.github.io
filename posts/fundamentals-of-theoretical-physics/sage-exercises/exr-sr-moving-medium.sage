# Exercise exr-sr-moving-medium (Blennow-Ohlsson, Problem 1.69): with n^2 = (1 - 2 w cos th + w^2)/w^2 the incident and
# refracted light have the same direction in the laboratory.
import numpy as np
def lab_dir(ux, uy, w):                                   # velocity in the slab frame -> direction in the laboratory
    g = 1/np.sqrt(1 - w*w); den = 1 - ux*w
    return np.array([(ux - w)/den, uy/(g*den)])
for w, th in [(0.3, 0.4), (0.2, 1.0), (0.45, 0.1)]:
    n = np.sqrt((1 - 2*w*np.cos(th) + w*w)/w**2); thr = np.arcsin(np.sin(th)/n)
    d1 = lab_dir(np.cos(th), np.sin(th), w); d2 = lab_dir(np.cos(thr)/n, np.sin(thr)/n, w)
    print("w = %.2f, theta' = %.2f: n = %.4f, angle between laboratory directions = %.2e rad" % (w, th, n, np.arccos(np.clip(d1 @ d2/np.linalg.norm(d1)/np.linalg.norm(d2), -1, 1))))
