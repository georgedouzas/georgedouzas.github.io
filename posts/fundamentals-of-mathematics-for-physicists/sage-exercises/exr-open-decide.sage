# Exercise exr-open-decide: a numerical test that the region v > u^2 is open: around sampled points, a small ball
# stays inside; around points of the parabola it does not.
import random
random.seed(int(1))
inside = lambda u, v: v > u**2
def has_ball(u, v, eps=1e-3, n=200):
    import math
    return all(inside(u + eps * math.cos(2 * math.pi * k / n), v + eps * math.sin(2 * math.pi * k / n)) for k in range(n))
pts = [(random.uniform(-1, 1), None) for _ in range(5)]
pts = [(u, u**2 + random.uniform(0.01, 1)) for u, _ in pts]
print("interior points have balls inside:", [has_ball(u, v) for u, v in pts])
print("boundary points do not:", [has_ball(u, u**2) for u in (-0.5, 0.0, 0.7)])
print("[PASS] sampled interior points have balls inside the region and boundary points do not:", all(has_ball(u, v) for u, v in pts) and not any(has_ball(u, u**2) for u in (-0.5, 0.0, 0.7)))
