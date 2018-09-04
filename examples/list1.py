import volumential.list1_gallery as lg

dim = 2
all_interactions = lg.generate_interactions(dimensions=dim)

# contains each sourcebox.center-to-targetbox.center vector
# source box is 4x4 to make all involved lengths to be integers
unique_interaction_vectors = set()
for (tbox, sbox) in all_interactions:
    unique_interaction_vectors.add(tuple(tbox.center - sbox.center))

list1_interactions = sorted(list(unique_interaction_vectors))
print(list1_interactions)

#vim: ft=pyopencl:fdm=marker
