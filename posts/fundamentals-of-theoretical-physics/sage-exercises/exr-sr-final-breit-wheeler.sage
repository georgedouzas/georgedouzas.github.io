# Exercise exr-sr-final-breit-wheeler: threshold of gamma gamma -> e+ e- for a head-on collision with a microwave
# background photon, and for other collision angles.
me = 0.51099895e6; eps = 6.3e-4
for th in [pi.n(), 2*pi.n()/3, pi.n()/2]:
    print("collision angle %.0f degrees: threshold %.3e eV" % (th*180/pi.n(), 2*me^2/(eps*(1 - cos(th)))))
