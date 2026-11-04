# Exercise exr-grp-orders-z12: the orders of the elements of Z_12 and the number of elements of each order.
R = Zmod(12)
orders = {k: R(k).additive_order() for k in range(12)}
print("orders:", orders)
from collections import Counter
cnt = Counter(orders.values())
print("number of elements of each order:", dict(sorted(cnt.items())))
print("each order divides 12:", all(12 % o == 0 for o in orders.values()))
print("order of k equals 12/gcd(k,12):", all(orders[k] == 12 // gcd(k, 12) for k in range(12)))
print("number of elements of order d equals Euler's phi(d):", all(cnt[d] == euler_phi(d) for d in divisors(12)))
