if plot_boxtree:
    exec(open("postprocessors/plot_boxtree.py").read())
    print("Boxtree plot generated.")

if write_vtu:
    exec(open("postprocessors/write_vtu.py").read())
    print("Solution written to vtu file")

if write_box_vtu:
    exec(open("postprocessors/write_box_vtu.py").read())
    print("Solution on box written to vtu file")
