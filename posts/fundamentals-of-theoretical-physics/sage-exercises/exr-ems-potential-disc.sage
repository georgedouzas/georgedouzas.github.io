# Exercise exr-ems-potential-disc (Pierrus, Question 2.5): potential on the axis of a uniformly charged disc, z > 0.
sigma, a, z, t = var('sigma a z t', domain='positive')
phi = (integrate(2 * pi * sigma * t / (4 * pi * sqrt(z^2 + t^2)), t, 0, a)).simplify_full()
print("phi(z) =", phi)
print("phi at the centre:", limit(phi, z=0))
Ez = -diff(phi, z)
print("-dphi/dz equals sigma/2 (1 - z/sqrt(z^2+a^2)):", bool((Ez - sigma / 2 * (1 - z / sqrt(z^2 + a^2))).simplify_full() == 0))
