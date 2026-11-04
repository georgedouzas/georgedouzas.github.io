# Exercise exr-homo-conj-classes-s4: the conjugacy classes of S_4 are its cycle types.
S4 = SymmetricGroup(4)
reps = S4.conjugacy_classes_representatives()
tot = 0
for r in reps:
    cl = S4.conjugacy_class(r)
    tot += len(cl)
    print("representative %-12s cycle type %-12s class size %2d" % (r, r.cycle_type(), len(cl)))
print("number of classes:", len(reps), "; sizes add up to", tot, "= |S_4|")
print("every element of a class has the same cycle type:", all(len({g.cycle_type() for g in S4.conjugacy_class(r)}) == 1 for r in reps))
print("number of partitions of 4:", Partitions(4).cardinality())
