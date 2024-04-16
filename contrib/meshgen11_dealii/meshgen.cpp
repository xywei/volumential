/* ---------------------------------------------------------------------
**
** Copyright (C) 2017 Xiaoyu Wei
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is
** furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
** OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
** THE SOFTWARE.
**
** -------------------------------------------------------------------*/

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

namespace py = pybind11;
namespace d = dealii;

constexpr double a = -1;
constexpr double b = 1;
const d::UpdateFlags update_flags =
    d::update_quadrature_points | d::update_JxW_values;

void greet() { std::cout << "Hello from mesh generation via deal.II." << std::endl; }

// {{{ utils

// make a copy of py::array to dealii::Vector
template <class Number>
d::Vector<Number> pyarr_to_dvec(py::array_t<Number> arr) {

  py::buffer_info info = arr.request();
  if (info.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be one");
  }

  // pointer to the data buffer
  Number *iter = (Number *)info.ptr;

  d::Vector<Number> dvec;
  dvec.reinit(info.shape[0]);
  Number *ptc = iter;
  for (unsigned int c = 0; c < info.shape[0]; ++c) {
    dvec[c] = *ptc;
    ++ptc;
  }

  return dvec;
}

template <int dim>
double l_infty_distance(d::Point<dim> &p1, d::Point<dim> &p2) {
  double dist = std::numeric_limits<double>::max();
  for (unsigned int d = 0; d < dim; ++d) {
    double tmp = std::abs(p1[d] - p2[d]);
    dist = tmp > dist ? dist : tmp;
  }
  return dist;
}

template <int dim>
double compute_q_point_radii(
    d::Point<dim> q_point,
    std::array<d::Point<dim>, d::GeometryInfo<dim>::vertices_per_cell>
        &vertices) {
  double rad = std::numeric_limits<double>::max();
  for (auto &&v : vertices) {
    double tmp = l_infty_distance(q_point, v);
    rad = tmp > rad ? rad : tmp;
  }
  return rad;
}

template <int dimension, int n_vertices>
d::Point<dimension>
compute_barycenter(std::array<d::Point<dimension>, n_vertices> &vertices) {

  d::Point<dimension> bc;
  for (unsigned int d = 0; d < dimension; ++d) {
    bc[d] = 0;
  }

  for (auto &&v : vertices) {
    bc += v;
  }
  bc /= n_vertices;
  return bc;
}

// }}}

// {{{ MeshGenerator class

template <int dimension> class MeshGenerator {
public:
  d::Triangulation<dimension> triangulation;
  d::FE_Q<dimension> fe;
  d::DoFHandler<dimension> dof_handler;
  d::QGauss<dimension> quadrature_formula;

  double box_a, box_b;

  unsigned int max_n_cells;
  unsigned int min_grid_level, max_grid_level;

  // q: quad order
  // level: initial (uniform) level
  // bounding box: [aa, bb]^dim
  MeshGenerator(
      int q, int level, double aa, double bb,
      py::args args, py::kwargs kwargs,
      unsigned int max_n_cells = std::numeric_limits<unsigned int>::max(),
      unsigned int min_grid_level = std::numeric_limits<unsigned int>::min(),
      unsigned int max_grid_level = std::numeric_limits<unsigned int>::max())
      : triangulation(d::Triangulation<dimension>::limit_level_difference_at_vertices), fe(q),
        dof_handler(triangulation), quadrature_formula(q), box_a(aa), box_b(bb),
        max_n_cells(max_n_cells), min_grid_level(min_grid_level),
        max_grid_level(max_grid_level) {
    d::GridGenerator::hyper_cube(triangulation, box_a, box_b);

    // initial number of levels must be compatible
    assert(level > min_grid_level);
    assert(level <= max_grid_level + 1);
    assert(std::pow(std::pow(2, dimension), level - 1) <= max_n_cells);

    if (level > 1) {
      triangulation.refine_global(level - 1);
    }
    this->dof_handler.distribute_dofs(fe);
  }

  // NOTE: Copy constructor is not supported since dealii::DoFHandler is not
  // copy constructable, but a dummy copy constructor is required by
  // boost.python to build the code.
  MeshGenerator(const MeshGenerator<dimension> &) : MeshGenerator(1, 1){};

  std::string greet() { return "Hello from MeshGen."; }

  void generate_gmsh(const std::string fn) {
    std::string filename = fn;
    std::ofstream output_file(filename);
    d::GridOut().write_msh(this->triangulation, output_file);
    std::cout << "Mesh written in " << filename << std::endl;
  }

  void write_vtu(const std::string fn) {
    std::string filename = fn;
    std::ofstream output_file(filename);
    d::GridOut().write_vtu(this->triangulation, output_file);
    std::cout << "Mesh written in " << filename << std::endl;
  }

  py::array get_q_points() {
    py::dtype dtype = py::dtype::of<double>();

    const size_t n_q_points = this->quadrature_formula.size();
    const size_t total_n_q_points = triangulation.n_active_cells() * n_q_points;

    std::array<size_t, 2> pt_shape = {total_n_q_points, dimension};
    std::array<size_t, 2> pt_strides = {dimension * sizeof(double),
                                        sizeof(double)};

    d::FEValues<dimension> fe_values(this->fe, this->quadrature_formula,
                                     d::update_quadrature_points);
    std::vector<double> points(pt_shape[0] * pt_shape[1], 0);
    auto ptp = points.data();

    auto cell = dof_handler.begin_active();
    auto endc = dof_handler.end();
    for (; cell != endc; ++cell) {
      fe_values.reinit(cell);
      std::vector<d::Point<dimension>> q_points =
          fe_values.get_quadrature_points();

      for (auto &&point : q_points) {
        for (unsigned int d = 0; d < dimension; ++d) {
          // For a preferred ordering of quad points
          // This does not change anything but the ordering due to the symmetry
          // in each direction (as long as the subdivided boxes are cubes), not
          // even weights are changed.
          *ptp = point[dimension - 1 - d];
          ++ptp;
        }
      }
    }

    return py::array(py::buffer_info(points.data(), sizeof(double),
                                     py::format_descriptor<double>::value, 2,
                                     pt_shape, pt_strides));
  }

  py::array get_q_weights() {
    py::dtype dtype = py::dtype::of<double>();

    const size_t n_q_points = this->quadrature_formula.size();
    const size_t total_n_q_points =
        this->triangulation.n_active_cells() * n_q_points;

    std::array<size_t, 1> w_shape = {total_n_q_points};
    std::array<size_t, 1> w_strides = {sizeof(double)};

    d::FEValues<dimension> fe_values(this->fe, this->quadrature_formula,
                                     d::update_JxW_values);

    std::vector<double> weights(w_shape[0], 0);
    auto ptw = weights.data();

    auto cell = this->dof_handler.begin_active();
    auto endc = this->dof_handler.end();
    for (; cell != endc; ++cell) {
      fe_values.reinit(cell);
      std::vector<double> q_weights;

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
        q_weights.push_back(fe_values.JxW(q_index));
      }

      for (auto &&weight : q_weights) {
        *ptw = weight;
        ++ptw;
      }
    }

    return py::array(py::buffer_info(weights.data(), sizeof(double),
                                     py::format_descriptor<double>::value, 1,
                                     w_shape, w_strides));
  }

  py::array get_cell_measures() {
    py::dtype dtype = py::dtype::of<double>();
    std::array<size_t, 1> m_shape = {this->triangulation.n_active_cells()};
    std::array<size_t, 1> m_strides = {sizeof(double)};
    std::vector<double> measures(m_shape[0], 0);
    auto ptm = measures.data();

    auto cell = this->dof_handler.begin_active();
    auto endc = this->dof_handler.end();
    for (; cell != endc; ++cell) {
      *ptm = cell->measure();
      ++ptm;
    }

    return py::array(py::buffer_info(measures.data(), sizeof(double),
                                     py::format_descriptor<double>::value, 1,
                                     m_shape, m_strides));
  }

  py::array get_cell_extents() {
    py::dtype dtype = py::dtype::of<double>();
    std::array<size_t, 1> m_shape = {this->triangulation.n_active_cells()};
    std::array<size_t, 1> m_strides = {sizeof(double)};
    std::vector<double> extents(m_shape[0], 0);
    auto ptm = extents.data();

    auto cell = this->dof_handler.begin_active();
    auto endc = this->dof_handler.end();
    for (; cell != endc; ++cell) {
      auto v0 = cell->vertex(0);
      auto v1 = cell->vertex(1);
      *ptm = l_infty_distance<dimension>(v0, v1);
      ++ptm;
    }

    return py::array(py::buffer_info(extents.data(), sizeof(double),
                                     py::format_descriptor<double>::value, 1,
                                     m_shape, m_strides));
  }

  py::array get_cell_centers() {
    py::dtype dtype = py::dtype::of<double>();
    std::array<size_t, 2> c_shape = {this->triangulation.n_active_cells(),
                                     dimension};
    std::array<size_t, 2> c_strides = {dimension * sizeof(double),
                                       sizeof(double)};
    std::vector<double> centers(c_shape[0] * c_shape[1], 0);
    auto ptc = centers.data();

    auto cell = this->dof_handler.begin_active();
    auto endc = this->dof_handler.end();
    for (; cell != endc; ++cell) {
      std::array<d::Point<dimension>,
                 d::GeometryInfo<dimension>::vertices_per_cell>
          vertices;
      for (unsigned int v = 0;
           v < d::GeometryInfo<dimension>::vertices_per_cell; ++v) {
        vertices[v] = cell->vertex(v);
      }
      auto barycenter = compute_barycenter<
          dimension, d::GeometryInfo<dimension>::vertices_per_cell>(vertices);
      for (unsigned int d = 0; d < dimension; ++d) {
        *ptc = barycenter[dimension - 1 - d];
        ++ptc;
      }
    }

    return py::array(py::buffer_info(centers.data(), sizeof(double),
                                     py::format_descriptor<double>::value, 2,
                                     c_shape, c_strides));
  }

  int n_active_cells() { return this->triangulation.n_active_cells(); }

  // Do both preparation for refinement and coarsening as well as mesh
  // smoothing. The function returns whether some cells' flagging has been
  // changed in the process.
  bool prepare_coarsening_and_refinement() {

    bool flagging_changed = false;

    if (this->triangulation.n_levels() > this->max_grid_level) {
      flagging_changed = true;
      for (const auto &cell :
           this->triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    }

    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      if (cell->coarsen_flag_set()) {
        flagging_changed = true;
        cell->clear_coarsen_flag();
      }

    return (flagging_changed &&
            this->triangulation.prepare_coarsening_and_refinement());
  }

  void execute_coarsening_and_refinement() {
    this->triangulation.execute_coarsening_and_refinement();
    this->dof_handler.distribute_dofs(fe);
  }

  // {{{ adaptive mesh refinement

  // (Legacy) driver function for mesh adaptivity
  int update_mesh(py::array_t<double> &criteria,
                  const double top_fraction_of_cells,
                  const double bottom_fraction_of_cells) {

    // calls refine_and_coarsen_fixed_number
    int n_active_cells = this->refine_and_coarsen_fixed_number(
        criteria, top_fraction_of_cells, bottom_fraction_of_cells);

    return n_active_cells;
  }

  // Interface for dealii::GridRefinement::refine_and_coarsen_fixed_number
  int refine_and_coarsen_fixed_number(py::array_t<double> &criteria,
                                      const double top_fraction_of_cells,
                                      const double bottom_fraction_of_cells) {

    auto ctr = pyarr_to_dvec<double>(criteria);

    d::GridRefinement::refine_and_coarsen_fixed_number(
        this->triangulation, ctr, top_fraction_of_cells,
        bottom_fraction_of_cells, this->max_n_cells);

    this->prepare_coarsening_and_refinement();
    this->execute_coarsening_and_refinement();

    return this->n_active_cells();
  }

  // Interface for dealii::GridRefinement::refine_and_coarsen_fixed_fraction
  int refine_and_coarsen_fixed_fraction(
      py::array_t<double> &criteria, const double top_fraction_of_errors,
      const double bottom_fraction_of_errors) {

    auto ctr = pyarr_to_dvec<double>(criteria);

    d::GridRefinement::refine_and_coarsen_fixed_fraction(
        this->triangulation, ctr, top_fraction_of_errors,
        bottom_fraction_of_errors, this->max_n_cells);

    this->prepare_coarsening_and_refinement();
    this->execute_coarsening_and_refinement();

    return this->n_active_cells();
  }

  // Interface for dealii::GridRefinement::refine_and_coarsen_optimize
  // NOTE: this function does not repsect max_n_cells!
  int refine_and_coarsen_optimize(py::array_t<double> &criteria,
                                  const unsigned int order) {

    auto ctr = pyarr_to_dvec<double>(criteria);

    d::GridRefinement::refine_and_coarsen_optimize(this->triangulation, ctr,
                                                   order);

    this->prepare_coarsening_and_refinement();
    this->execute_coarsening_and_refinement();

    if (this->n_active_cells() > this->max_n_cells) {
      std::cout << "Warning: max_n_cells has been exceeded!" << std::endl;
    }

    return this->n_active_cells();
  }

  // Interface for dealii::GridRefinement::refine
  // NOTE: this function does not repsect max_n_cells!
  int refine(py::array_t<double> &criteria, const double threshold,
             const unsigned int max_to_mark =
                 std::numeric_limits<unsigned int>::max()) {

    auto ctr = pyarr_to_dvec<double>(criteria);

    d::GridRefinement::refine(this->triangulation, ctr, threshold, max_to_mark);

    this->prepare_coarsening_and_refinement();
    this->execute_coarsening_and_refinement();

    if (this->n_active_cells() > this->max_n_cells) {
      std::cout << "Warning: max_n_cells has been exceeded!" << std::endl;
    }

    return this->n_active_cells();
  }

  // Interface for dealii::GridRefinement::coarsen
  int coarsen(py::array_t<double> &criteria, const double threshold) {

    auto ctr = pyarr_to_dvec<double>(criteria);

    d::GridRefinement::coarsen(this->triangulation, ctr, threshold);

    this->prepare_coarsening_and_refinement();
    this->execute_coarsening_and_refinement();

    return this->n_active_cells();
  }

  // }}}

  // Show some info about the mesh
  void print_info() {

    std::cout << "Number of cells: " << this->triangulation.n_cells()
              << std::endl;

    std::cout << "Number of active cells: "
              << this->triangulation.n_active_cells() << std::endl;

    d::FEValues<dimension> fe_values(fe, quadrature_formula,
                                     d::update_quadrature_points);
    const unsigned int n_q_points = quadrature_formula.size();
    std::cout << "Number of quad points per cell: " << n_q_points << std::endl;
  }
};
typedef MeshGenerator<1> MeshGen1D;
typedef MeshGenerator<2> MeshGen2D;
typedef MeshGenerator<3> MeshGen3D;

// }}}

// {{{ (Legacy) make uniform mesh
// A legacy one-stop mesh generation function, returns quad points, weights
// and some spacing info within a tuple.
template <int dim> py::tuple make_uniform_cubic_grid_details(int q, int level) {
  py::dtype dtype = py::dtype::of<double>();

  d::Triangulation<dim> triangulation;
  d::FE_Q<dim> fe(q);
  d::DoFHandler<dim> dof_handler(triangulation);
  d::QGauss<dim> quadrature_formula(q);

  d::GridGenerator::hyper_cube(triangulation, a, b);
  if (level > 1) {
    triangulation.refine_global(level - 1);
  }
  dof_handler.distribute_dofs(fe);
  // std::cout << "Number of active cells: " << triangulation.n_active_cells()
  //<< std::endl;

  d::FEValues<dim> fe_values(fe, quadrature_formula, update_flags);
  const size_t n_q_points = quadrature_formula.size();
  // std::cout << "Number of quad points per cell: " << n_q_points << std::endl;

  const size_t total_n_q_points = triangulation.n_active_cells() * n_q_points;

  std::array<size_t, 2> pt_shape = {total_n_q_points, dim};
  std::array<size_t, 2> pt_strides = {dim * sizeof(double), sizeof(double)};
  std::array<size_t, 1> w_shape = {total_n_q_points};
  std::array<size_t, 1> w_strides = {sizeof(double)};

  // Quad points
  std::vector<double> points(pt_shape[0] * pt_shape[1], 0);
  // Quad weights
  std::vector<double> weights(w_shape[0], 0);
  // Distance (l_infty) to the closest cell vertex
  // (used for reconstructing the mesh in boxtree)
  std::vector<double> radii(w_shape[0], 0);
  auto ptp = points.data();
  auto ptw = weights.data();
  auto ptr = radii.data();

  // For margins
  std::array<double, dim * 2> margins;
  std::array<double *, dim * 2> mpts;
  margins.fill(std::numeric_limits<double>::max());

  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
    fe_values.reinit(cell);
    std::vector<d::Point<dim>> q_points = fe_values.get_quadrature_points();
    std::vector<double> q_weights;
    std::array<d::Point<dim>, d::GeometryInfo<dim>::vertices_per_cell> vertices;

    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
      q_weights.push_back(fe_values.JxW(q_index));
    }

    for (unsigned int v = 0; v < d::GeometryInfo<dim>::vertices_per_cell; ++v) {
      vertices[v] = cell->vertex(v);
    }

    for (auto &&point : q_points) {
      for (unsigned int d = 0; d < dim; ++d) {
        // For a preferred ordering of quad points
        // This does not change anything but the ordering due to the symmetry
        // in each direction (as long as the subdivided boxes are cubes), not
        // even weights are changed.
        *ptp = point[dim - 1 - d];
        ++ptp;
      }

      // gather some info about margins
      for (unsigned int d = 0; d < dim; ++d) {
        auto margin_a_id = d * 2;
        auto margin_b_id = d * 2 + 1;
        if (std::abs(point[d] - a) < margins[margin_a_id]) {
          margins[margin_a_id] = std::abs(point[d] - a);
          mpts[margin_a_id] = ptr;
        }
        if (std::abs(point[d] - b) < margins[margin_b_id]) {
          margins[margin_b_id] = std::abs(point[d] - b);
          mpts[margin_b_id] = ptr;
        }
      }

      ++ptr;
    }

    for (auto &&weight : q_weights) {
      *ptw = weight;
      ++ptw;
    }
  }

  for (unsigned int i = 0; i < dim * 2; ++i) {
    *(mpts[i]) = margins[i];
  }

  py::tuple result = py::make_tuple(
      py::array(py::buffer_info(points.data(), sizeof(double),
                                py::format_descriptor<double>::value, 2,
                                pt_shape, pt_strides)),
      py::array(py::buffer_info(weights.data(), sizeof(double),
                                py::format_descriptor<double>::value, 1,
                                w_shape, w_strides)),
      py::array(py::buffer_info(radii.data(), sizeof(double),
                                py::format_descriptor<double>::value, 1,
                                w_shape, w_strides)));

  return result;
}

py::tuple make_uniform_cubic_grid(int q, int level, int dim,
                                  py::args args, py::kwargs kwargs) {
  if (dim == 1) {
    return make_uniform_cubic_grid_details<1>(q, level);
  } else if (dim == 2) {
    return make_uniform_cubic_grid_details<2>(q, level);
  } else if (dim == 3) {
    return make_uniform_cubic_grid_details<3>(q, level);
  } else {
    std::cout << "Dimension must be 1,2 or 3." << std::endl;
    return py::tuple();
  }
}

// }}}

PYBIND11_MODULE(meshgen_dealii, m) {
  m.doc() = "A mesh generator for volumential.";

  m.def("greet", &greet, "Greetings! This is meshgen11.");

  m.def("make_uniform_cubic_grid", &make_uniform_cubic_grid,
        "Make a simple grid", py::arg("degree"), py::arg("level") = 1,
        py::arg("dim") = 2);

  py::class_<MeshGen1D>(m, "MeshGen1D")
      .def(py::init<int, int, double, double, py::args, py::kwargs>(), py::arg("degree"),
           py::arg("level"), py::arg("a"), py::arg("b"))
      .def("greet", &MeshGen1D::greet)
      .def("get_q_points", &MeshGen1D::get_q_points)
      .def("get_q_weights", &MeshGen1D::get_q_weights)
      .def("get_cell_centers", &MeshGen1D::get_cell_centers)
      .def("get_cell_extents", &MeshGen1D::get_cell_extents)
      .def("get_cell_measures", &MeshGen1D::get_cell_measures)
      .def("n_active_cells", &MeshGen1D::n_active_cells)
      .def("prepare_coarsening_and_refinement",
           &MeshGen1D::prepare_coarsening_and_refinement)
      .def("execute_coarsening_and_refinement",
           &MeshGen1D::execute_coarsening_and_refinement)
      .def("update_mesh", &MeshGen1D::update_mesh)
      .def("refine_and_coarsen_fixed_number",
           &MeshGen1D::refine_and_coarsen_fixed_number)
      .def("refine_and_coarsen_fixed_fraction",
           &MeshGen1D::refine_and_coarsen_fixed_fraction)
      .def("refine_and_coarsen_optimize",
           &MeshGen1D::refine_and_coarsen_optimize)
      .def("refine", &MeshGen1D::refine)
      .def("coarsen", &MeshGen1D::coarsen)
      .def("print_info", &MeshGen1D::print_info)
      .def("generate_gmsh", &MeshGen1D::generate_gmsh)
      .def("write_vtu", &MeshGen1D::write_vtu);

  py::class_<MeshGen2D>(m, "MeshGen2D")
      .def(py::init<int, int, double, double, py::args, py::kwargs>(), py::arg("degree"),
           py::arg("level"), py::arg("a"), py::arg("b"))
      .def("greet", &MeshGen2D::greet)
      .def("get_q_points", &MeshGen2D::get_q_points)
      .def("get_q_weights", &MeshGen2D::get_q_weights)
      .def("get_cell_centers", &MeshGen2D::get_cell_centers)
      .def("get_cell_extents", &MeshGen2D::get_cell_extents)
      .def("get_cell_measures", &MeshGen2D::get_cell_measures)
      .def("n_active_cells", &MeshGen2D::n_active_cells)
      .def("prepare_coarsening_and_refinement",
           &MeshGen2D::prepare_coarsening_and_refinement)
      .def("execute_coarsening_and_refinement",
           &MeshGen2D::execute_coarsening_and_refinement)
      .def("update_mesh", &MeshGen2D::update_mesh)
      .def("refine_and_coarsen_fixed_number",
           &MeshGen2D::refine_and_coarsen_fixed_number)
      .def("refine_and_coarsen_fixed_fraction",
           &MeshGen2D::refine_and_coarsen_fixed_fraction)
      .def("refine_and_coarsen_optimize",
           &MeshGen2D::refine_and_coarsen_optimize)
      .def("refine", &MeshGen2D::refine)
      .def("coarsen", &MeshGen2D::coarsen)
      .def("print_info", &MeshGen2D::print_info)
      .def("generate_gmsh", &MeshGen2D::generate_gmsh)
      .def("write_vtu", &MeshGen2D::write_vtu);

  py::class_<MeshGen3D>(m, "MeshGen3D")
      .def(py::init<int, int, double, double, py::args, py::kwargs>(), py::arg("degree"),
           py::arg("level"), py::arg("a"), py::arg("b"))
      .def("greet", &MeshGen3D::greet)
      .def("get_q_points", &MeshGen3D::get_q_points)
      .def("get_q_weights", &MeshGen3D::get_q_weights)
      .def("get_cell_centers", &MeshGen3D::get_cell_centers)
      .def("get_cell_extents", &MeshGen3D::get_cell_extents)
      .def("get_cell_measures", &MeshGen3D::get_cell_measures)
      .def("n_active_cells", &MeshGen3D::n_active_cells)
      .def("prepare_coarsening_and_refinement",
           &MeshGen3D::prepare_coarsening_and_refinement)
      .def("execute_coarsening_and_refinement",
           &MeshGen3D::execute_coarsening_and_refinement)
      .def("update_mesh", &MeshGen3D::update_mesh)
      .def("refine_and_coarsen_fixed_number",
           &MeshGen3D::refine_and_coarsen_fixed_number)
      .def("refine_and_coarsen_fixed_fraction",
           &MeshGen3D::refine_and_coarsen_fixed_fraction)
      .def("refine_and_coarsen_optimize",
           &MeshGen3D::refine_and_coarsen_optimize)
      .def("refine", &MeshGen3D::refine)
      .def("coarsen", &MeshGen3D::coarsen)
      .def("print_info", &MeshGen3D::print_info)
      .def("generate_gmsh", &MeshGen3D::generate_gmsh)
      .def("write_vtu", &MeshGen3D::write_vtu);
}
