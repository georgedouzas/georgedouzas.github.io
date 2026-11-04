# Exercise exr-fan-mazur: the l^p norm of the average of the first N unit vectors is N^(1/p - 1).
for p in (1, 1.5, 2, 4):
    print("p = %3s:" % p, [round(vector([1.0 / N] * N).norm(p), 5) for N in (1, 10, 100, 1000)], " N^(1/p - 1) at N = 1000:", round(1000^(1 / p - 1), 5))
