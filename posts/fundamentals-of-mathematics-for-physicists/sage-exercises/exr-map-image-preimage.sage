# Exercise exr-map-image-preimage: image and preimages of a finite map given as a dictionary.
f = {1: "a", 2: "c", 3: "a", 4: "a"}
image = set(f.values())
preimage = lambda V: {x for x in f if f[x] in V}
print("image:", sorted(image))
for V in ({"a"}, {"b"}, {"a", "c"}): print("preimage of", sorted(V), ":", sorted(preimage(V)))
print("[PASS] image {a, c}; preimages {1,3,4}, {}, {1,2,3,4}:", image == {"a", "c"} and preimage({"a"}) == {1, 3, 4} and preimage({"b"}) == set() and preimage({"a", "c"}) == {1, 2, 3, 4})
