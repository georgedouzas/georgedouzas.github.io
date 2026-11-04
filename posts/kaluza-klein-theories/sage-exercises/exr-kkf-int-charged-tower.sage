# Exercise exr-kkf-int-charged-tower: masses |n + 1/2 + theta|/R of an antiperiodic fermion with a Wilson line theta of a
# five-dimensional U(1), below 3/R, with multiplicities, for theta = 0, 1/4, 1/2.
from collections import Counter
for th in (0, QQ(1) / 4, QQ(1) / 2):
    ms = Counter(abs(n + QQ(1) / 2 + th) for n in range(-10, 10) if abs(n + QQ(1) / 2 + th) < 3)
    print("theta = %s: masses (units 1/R) with multiplicity:" % th, sorted(ms.items()), "; massless mode:", 0 in ms)
