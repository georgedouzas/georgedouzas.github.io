# Exercise exr-ems-dipole-offaxis: numerical Biot-Savart field of a unit loop (I = 1, radius 1) against the dipole field.
import numpy as np
def B_loop(P, N=4000):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False); dt = 2 * np.pi / N
    xs = np.stack([np.cos(t), np.sin(t), 0 * t], axis=1); dl = np.stack([-np.sin(t), np.cos(t), 0 * t], axis=1) * dt
    R = P - xs; Rn = np.linalg.norm(R, axis=1)[:, None]
    return (np.cross(dl, R) / Rn**3).sum(axis=0) / (4 * np.pi)
def B_dip(P, m=np.array([0, 0, np.pi])):
    r = np.linalg.norm(P); n = P / r
    return (3 * np.dot(m, n) * n - m) / (4 * np.pi * r**3)
for r in [5.0, 10.0, 20.0]:
    th = np.pi / 3; P = r * np.array([np.sin(th), 0, np.cos(th)])
    b, d = B_loop(P), B_dip(P)
    print("r = %4.1f: loop %s, dipole %s, relative error %.2e" % (r, np.array2string(b, precision=6), np.array2string(d, precision=6), np.linalg.norm(b - d) / np.linalg.norm(d)))
