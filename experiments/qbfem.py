import numpy as np
import sys
import gmsh
import pyopencl as cl
import os
from pytools.obj_array import make_obj_array
import matplotlib.pyplot as plt

from dolfin import *

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

source_mesh_order = 2  # order for direct quadratures over the source mesh
vp_solu_degree = 2   # degree of FE space for solution
quad_degree = vp_solu_degree  # degree of FE space for rhs

# {{{ load list of coordinates from a gmsh file

def load_list_data(source_mesh_file, h=0.05):
    from meshmode.mesh.io import read_gmsh
    mesh = read_gmsh(source_mesh_file)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
        PolynomialWarpAndBlendGroupFactory
    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    discr = Discretization(cl_ctx, mesh,
                           PolynomialWarpAndBlendGroupFactory(order=1))
    nodes = discr.nodes().with_queue(queue)
    coordx = nodes[0].get()
    coordy = nodes[1].get()

    list_data = []
    for (x, y) in zip(coordx, coordy):
        list_data.append(x)
        list_data.append(y)
        list_data.append(0)
        list_data.append(h)

    return list_data

# }}} End load list of coordinates from a gmsh file


# {{{ setup helpers


def gmsh_setup():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    # meshmode does not support other versions
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    # FEniCS does not support 2nd order mesh
    gmsh.option.setNumber("Mesh.ElementOrder", 1)


def gmsh_setup_higher_order(order=2):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    # meshmode does not support other versions
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)


def gmsh_add_model_unit_square(lc=1, name="square", pos=np.array([0, 0])):
    gmsh.model.add(name)
    x0, y0 = pos
    gmsh.model.geo.addPoint(x0 + 0, y0 + 0, 0, lc, 1)
    gmsh.model.geo.addPoint(x0 + 1, y0 + 0,  0, lc, 2)
    gmsh.model.geo.addPoint(x0 + 1, y0 + 1, 0, lc, 3)
    gmsh.model.geo.addPoint(x0 + 0, y0 + 1, 0, lc, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(3, 2, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.addPhysicalGroup(0, [1, 2], 1)
    gmsh.model.addPhysicalGroup(1, [1, 2], 2)
    gmsh.model.addPhysicalGroup(2, [1], 6)

    gmsh.model.geo.synchronize()


def make_mesh(h=0.01):
    gmsh_setup()
    tag_box = gmsh.model.occ.addRectangle(x=0, y=0, z=0, dx=1, dy=1, tag=-1)
    tag_disk = gmsh.model.occ.addDisk(xc=1, yc=0, zc=0, rx=0.8, ry=0.5, tag=-1)
    dimtags_ints, dimtags_map_ints = gmsh.model.occ.intersect(
        objectDimTags=[(2, tag_box)],
        toolDimTags=[(2, tag_disk)],
        tag=-1, removeObject=True, removeTool=True)
    tag_outer_box = gmsh.model.occ.addRectangle(
            x=-2, y=-2, z=0, dx=5, dy=5, tag=-1)
    tag_ext = gmsh.model.occ.cut(
            objectDimTags=[(2, tag_outer_box)],
            toolDimTags=dimtags_ints,
            tag=-1, removeObject=True, removeTool=False)

    gmsh.model.occ.synchronize()

    # Use f1 to refine center box
    f1 = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f1, "VIn", 0.5)
    gmsh.model.mesh.field.setNumber(f1, "VOut", 1)
    gmsh.model.mesh.field.setNumber(f1, "XMin", 0)
    gmsh.model.mesh.field.setNumber(f1, "XMax", 1)
    gmsh.model.mesh.field.setNumber(f1, "YMin", 0)
    gmsh.model.mesh.field.setNumber(f1, "YMax", 1)

    # Use f2 to refine (user defined) region

    if os.path.isfile("src_mesh.msh"):
        print("Using src_mesh to refine the mesh.")
        list_data = load_list_data("src_mesh.msh", h=h)
        v1 = gmsh.view.add(name="mesh_density")
        # gmsh points are always 3D
        # list data format: [xi, yi, zi, val_i for i in range(N)]
        gmsh.view.addListData(v1, "SP", len(list_data)//4, list_data)
        f2 = gmsh.model.mesh.field.add(fieldType="PostView")
        gmsh.model.mesh.field.setNumber(f2, "ViewTag", v1)
    else:
        f2 = gmsh.model.mesh.field.add("MatEval")
        gmsh.model.mesh.field.setString(f2, "F0", "1")

    f3 = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f3, "FieldsList", [f1, f2])

    gmsh.model.mesh.field.setAsBackgroundMesh(f3)

    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
    gmsh.finalize()


def make_source_mesh(order=2):
    gmsh_setup_higher_order(order=order)
    tag_box = gmsh.model.occ.addRectangle(x=0, y=0, z=0, dx=1, dy=1, tag=-1)
    tag_disk = gmsh.model.occ.addDisk(xc=1, yc=0, zc=0, rx=0.8, ry=0.5, tag=-1)
    dimtags_ints, dimtags_map_ints = gmsh.model.occ.intersect(
        objectDimTags=[(2, tag_box)],
        toolDimTags=[(2, tag_disk)],
        tag=-1, removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("src_mesh.msh")
    gmsh.finalize()


# }}} End setup helpers


# {{{ visualizers

def visualize_mesh_geometry(msh_file_path,
                            out_file_path="geometry.vtu",
                            vis_order=1):
    from meshmode.mesh.io import read_gmsh
    mesh = read_gmsh(msh_file_path)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
        PolynomialWarpAndBlendGroupFactory
    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    discr = Discretization(cl_ctx, mesh,
                           PolynomialWarpAndBlendGroupFactory(order=1))
    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(queue, discr, vis_order=vis_order)
    if os.path.isfile(out_file_path):
        os.remove(out_file_path)
    vis.write_vtk_file(out_file_path,
                       [
                           ("f", discr.nodes()[0]),
                           ])


def plot_box(x=0, y=0, dx=1, dy=1, **kwargs):
    plt.plot([x, x + dx, x + dx, x, x],
             [y, y, y + dy, y + dy, y], **kwargs)


def plot_list1_boxes(x=0, y=0, dx=1, dy=1, **kwargs):
    plot_box(x-dx, y-dy, dx, dy, **kwargs)
    plot_box(x-dx, y, dx, dy, **kwargs)
    plot_box(x-dx, y+dy, dx, dy, **kwargs)
    plot_box(x, y-dy, dx, dy, **kwargs)
    plot_box(x, y, dx, dy, **kwargs)
    plot_box(x, y+dy, dx, dy, **kwargs)
    plot_box(x+dx, y-dy, dx, dy, **kwargs)
    plot_box(x+dy, y, dx, dy, **kwargs)
    plot_box(x+dy, y+dy, dx, dy, **kwargs)


def plot_vtk_2d(vtk_file_path, fill=False, **kwargs):
    # conda install vtk
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    data = reader.GetOutput()

    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())

    triangles=  vtk_to_numpy(data.GetCells().GetData())
    ntri = triangles.size//4  # number of cells
    tri = np.take(triangles,[n for n in range(triangles.size) if n%4 != 0]).reshape(ntri,3)

    n_arrays = reader.GetNumberOfPointArrays()
    if 0:
        for i in range(n_arrays):
            print("ArrayName in vtk file:", reader.GetPointArrayName(i))

    if n_arrays > 0:
        u = vtk_to_numpy(data.GetPointData().GetArray(0))

    if fill:
        plt.tripcolor(x[:,0], x[:,1], tri, **kwargs)
    else:
        plt.triplot(x[:,0], x[:,1], tri, **kwargs)
    plt.gca().set_aspect('equal')

    return x

# }}} End visualizers


make_source_mesh(order=source_mesh_order)
make_mesh(h=1)
visualize_mesh_geometry("mesh.msh")
visualize_mesh_geometry("src_mesh.msh", out_file_path="source_geometry.vtu")

os.system("dolfin-convert mesh.msh mesh.xml")

figs = []

# {{{ plot meshes

if 1:
    # plot source mesh
    figs.append(plt.figure(figsize=(8, 8)))
    plot_vtk_2d("source_geometry.vtu", color='black', alpha=1, linewidth=0.5)
    plot_box(0, 0, 1, 1, color='black', alpha=1, linewidth=0.5)
    if 0:
        plot_list1_boxes(0, 0, 1, 1)
    plt.axis('off')
    plt.title("Source mesh")

if 1:
    # plot FEM mesh
    figs.append(plt.figure(figsize=(8, 8)))
    plot_vtk_2d("geometry.vtu")
    plot_list1_boxes(0, 0, 1, 1)
    plt.axis('off')
    plt.title("FEM mesh")

# }}} End plot meshes

mesh = Mesh("mesh.xml")
V = FunctionSpace(mesh, "Lagrange", vp_solu_degree)
FE = FiniteElement(family='Lagrange', cell='triangle', degree=vp_solu_degree)

# Avoid evaluating f on the cell boundary by using 'quadrature' elements
# "default" uses boundary nodes!
# "canonical" does not (probably that is what is reffered to as 'collapsed' Gauss quadrature.)
QFE = FiniteElement('Quadrature', mesh.ufl_cell(), degree=quad_degree, quad_scheme="canonical")

def source_mask(x):
    if (x[0] > 0 and x[0] < 1 and x[1] > 0 and x[1] < 1 and
      (x[0]-1)*(x[0]-1) / 0.64 + (x[1])*(x[1]) / 0.25 < 1):
        return 1
    else:
        return 0
    
def near_jump(x):
    return 1

near_jump_nodes_x = []
near_jump_nodes_y = []

class SourceExp(UserExpression):
    def eval(self, val, x):
        if near_jump(x):
            near_jump_nodes_x.append(x[0])
            near_jump_nodes_y.append(x[1])
        val[0] = source_mask(x)

f = SourceExp(element=QFE)

from sumpy.kernel import LaplaceKernel
from pymbolic import evaluate
knl = LaplaceKernel(2)

vp_quad_order = source_mesh_order


def eval_knl(rvec):
    knl_val = evaluate(knl.expression,
                       context={
                           "log": np.log,
                           "sqrt": np.sqrt,
                           "d": rvec})
    knl_scal = evaluate(knl.global_scaling_const,
                        context={
                                 "pi": np.pi
                                })
    return knl_val * knl_scal


from meshmode.mesh.io import read_gmsh
source_mesh = read_gmsh("src_mesh.msh")

from pytential import bind, sym
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
    InterpolatoryQuadratureSimplexGroupFactory
discr = Discretization(
    cl_ctx, source_mesh,
    InterpolatoryQuadratureSimplexGroupFactory(vp_quad_order))
nodes = discr.nodes().with_queue(queue)
weights = discr.quad_weights(queue).with_queue(queue)
src_area = bind(discr, sym.integral(2, 2, 1))(queue)

# error in computing the area
print("Error in computing the area = %E" %
      np.linalg.norm(src_area - np.pi * 0.8 * 0.5 / 4))

quad_scale = src_area / np.sum(weights)


def source_density(x):
    # The source density, to be interpolated from physical mesh.
    # To evaluate derivatives of VP, simply use derivatives of
    # source density.
    return 0 * (x[0] + x[1]) + 1


source_vals = source_density(nodes)
sym_integrand = sym.var("source_density") * sym.var("translated_knl")


def vp_direct_quadrature(target):
    assert len(target) == 2

    rvec = nodes.get()
    rvec[0] = target[0] - rvec[0]
    rvec[1] = target[1] - rvec[1]
    knl_vals = cl.array.to_device(queue, eval_knl(rvec))

    return bind(discr, sym.integral(2, 2, sym_integrand))(
        queue,
        source_density=source_density(nodes),
        translated_knl=knl_vals
        )

# Get mesh info and manually eval bcs

# Weights are scaled cell-wise with area_elements (Jacobians)
area_elements = bind(discr, sym.area_element(2, 2))(queue)
JxW = area_elements * weights

# JxW sums to the area of the domain
print("Error of Jxw = %E" % (np.sum(JxW.get()) - src_area))


def eval_vp_direct_quadrature(targets):

    ntgts = len(targets)

    rvecs = []
    rvec_base = nodes.get()
    for t in targets:
        rvec = rvec_base.copy()
        rvec[0] = t[0] - rvec[0]
        rvec[1] = t[1] - rvec[1]
        rvecs.append(rvec)

    # Not optimized, but this is not the bottleneck.
    knl_vals = make_obj_array([
        cl.array.to_device(queue, eval_knl(rvec))
        for rvec in rvecs
    ])

    return np.array(
        [
            (cl.array.sum(
                source_vals * kval * JxW)).get()
            for kval in knl_vals ]
    ).reshape(-1)


# Check correctness
# np.linalg.norm(eval_vp_direct_quadrature(targets) - bvals2, ord=np.Inf)
# print(eval_vp_direct_quadrature(targets))

# Define Dirichlet boundary (all of the boundary)
# Return 1 on the boundary (must return 0 on all interior points)
def boundary(x):
    return (x[0] > 3 - DOLFIN_EPS or
            x[0] < -2 + DOLFIN_EPS or
            x[1] > 3 - DOLFIN_EPS or
            x[1] < -2 + DOLFIN_EPS)

# Define boundary condition
bdry_nodes = []
cur_bdry_node_id = 0
class BCExp(UserExpression):
    def eval(self, val, x):
        # TODO: eval bcs from quadrature
        # Computes vp from the source region evaluated at target x
        # The quadrature is computed from pytential for its better support for higher order meshes
        val[0] =  next(bdry_value_iter)
        #val[0] = vp_direct_quadrature(target=x)
                
class BCProbExp(UserExpression):
    def eval(self, val, x):
        # Prob for boundary points.
        bdry_nodes.append(x.copy())
        val[0] = 23.3
        

u1 = BCProbExp(element=FE)
bcprob = DirichletBC(V, u1, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v))*dx
L = f*v*dx(metadata={'quadrature_degree': quad_degree, 'quadrature_rule': 'canonical'})

mat = assemble(a)
rhs = assemble(L)
dummy_rhs = rhs.copy()

# Prob for boundary nodes
bdry_nodes = []
bcprob.apply(mat, dummy_rhs)
#print(bdry_nodes)

bdry_vals = eval_vp_direct_quadrature(bdry_nodes)
#print(bdry_vals)

bdry_value_iter = iter(bdry_vals)
u0 = BCExp(element=FE)
bc = DirichletBC(V, u0, boundary)
bc.apply(mat, rhs)

# Compute solution
u = Function(V)
#solve(a == L, u, bc)
solve(mat, u.vector(), rhs)


import matplotlib.tri as tri

triang = tri.Triangulation(*mesh.coordinates().reshape((-1, 2)).T,
                           triangles=mesh.cells())
Z = u.compute_vertex_values(mesh)

figs.append(plt.figure(figsize=(8,8)))
plt.tricontourf(triang, Z, 20)
plt.colorbar()
plt.axis("square")
plt.title("Solution Contour Plot")

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

if 1:
    print("Generating plots on a refined mesh. Could take some time..")

    fine_mesh = refine(refine(refine(refine(mesh))))
    V_fine = FunctionSpace(fine_mesh, "DG", 2)

    triang = tri.Triangulation(*fine_mesh.coordinates().reshape((-1, 2)).T,
                               triangles=fine_mesh.cells())

    uu = interpolate(u, V_fine)
    Z = uu.compute_vertex_values(fine_mesh)
    ZZ = Z.copy()
    print(uu(0,1))
    Z[Z<uu(0,1)] = 0

    figs.append(plt.figure(figsize=(8, 8)))
    ff = interpolate(f, V_fine)
    ZF = ff.compute_vertex_values(fine_mesh)
    plt.tricontourf(triang, ZF, 200)
    plt.colorbar()
    plt.axis("square")
    plot_vtk_2d("source_geometry.vtu", color='black', alpha=1, linewidth=0.5)
    plt.axis([0, 1, 0, 1])
    plt.axis('off')
    plt.title("Source")

    figs.append(plt.figure(figsize=(8, 8)))
    ax = plt.gca()
    im = ax.tricontourf(triang, Z, 200, linewidth=0)
    divider = make_axes_locatable(ax)
    plt.axis("square")
    plot_vtk_2d("source_geometry.vtu", color='black', alpha=1, linewidth=0.5)
    plot_box(0, 0, 1, 1, color='black', alpha=1, linewidth=0.5)
    # plot_list1_boxes(0, 0, 1, 1)
    plt.axis([0, 1, 0, 1])
    plt.axis('off')
    plt.title("Solution over source box")

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        # return r'${} \times 10^{{{}}}$'.format(a, b)
        return r'${}$'.format(a)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, format=ticker.FuncFormatter(fmt))

    cbar.set_label(r'Volume Potential ($\times 10^{-2}$)', rotation=90)
    plt.tight_layout()

    figs.append(plt.figure(figsize=(8, 8)))
    ax = plt.gca()
    im = ax.tricontourf(triang, ZZ, 200, linewidth=0)
    divider = make_axes_locatable(ax)
    plt.axis("square")
    plot_vtk_2d("source_geometry.vtu", color='black', alpha=1, linewidth=0.5)
    plot_box(0, 0, 1, 1, color='black', alpha=1, linewidth=0.5)
    plot_list1_boxes(0, 0, 1, 1)
    # plt.axis([0, 1, 0, 1])
    plt.axis('off')
    plt.title("Solution over list1")

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        # return r'${} \times 10^{{{}}}$'.format(a, b)
        return r'${}$'.format(a)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, format=ticker.FuncFormatter(fmt))

    cbar.set_label(r'Volume Potential ($\times 10^{-2}$)', rotation=90)
    plt.tight_layout()

if 0:
    # Show plots
    plt.show()
else:
    # save plots to png
    ifig = 1
    for fig in figs:
        fig.savefig("fig-%d.png" % ifig)
        ifig += 1


