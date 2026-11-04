# Exercise exr-grb-rocket-signal (Blennow-Ohlsson, Problem 2.126): a spaceship at 1000 km altitude moving radially
# upward at 0.1 km/s sends a light signal to the ground; the gravitational shift (blue, the light falls into the
# potential well) and the Doppler shift (red, the source recedes) to lowest order, with the book's data.
G, M, c = 6.67e-11, 5.98e24, 3.0e8
R, h, v = 6.3e6, 1.0e6, 1.0e2
grav = G * M / c**2 * (1 / R - 1 / (R + h))      # nu_received/nu_emitted - 1 from the potential difference
dopp = -v / c                                     # first-order longitudinal Doppler shift of a receding source
print("gravitational shift  (Delta nu/nu): %+.3e" % grav)
print("Doppler shift        (Delta nu/nu): %+.3e" % dopp)
print("total                (Delta nu/nu): %+.3e" % (grav + dopp))
print("Doppler/gravitational ratio: %.0f" % abs(dopp / grav))
