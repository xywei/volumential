# Boxtree
import matplotlib.pyplot as plt

if dim == 2:
    plt.plot(q_points[0].get(), q_points[1].get(), ".")

from boxtree.visualization import TreePlotter
plotter = TreePlotter(tree.get(queue=queue))
plotter.draw_tree(fill=False, edgecolor="black")
#plotter.draw_box_numbers()
plotter.set_bounding_box()
plt.gca().set_aspect("equal")

# plt.draw()
# plt.show()
plt.savefig("tree.png")
