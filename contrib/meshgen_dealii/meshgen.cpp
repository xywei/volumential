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

#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <array>
#include <cmath>

#include <boost/python/def.hpp>
#include <boost/python/docstring_options.hpp>
#include <boost/python/module.hpp>
#include <boost/python/numpy.hpp>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

namespace p  = boost::python;
namespace np = boost::python::numpy;
namespace d  = dealii;

constexpr int        dim = 2;
constexpr double     a   = -1;
constexpr double     b   = 1;
const d::UpdateFlags update_flags =
    d::update_quadrature_points | d::update_JxW_values;

void greet() { std::cout << "Yey!" << std::endl; }

double l_infty_distance(d::Point<dim> & p1, d::Point<dim> & p2){
  double dist = std::numeric_limits<double>::max();
  for (unsigned int d=0; d < dim; ++d) {
    double tmp = std::abs(p1[d] - p2[d]);
    dist = tmp > dist ? dist : tmp;
  }
  return dist;
}

double compute_q_point_radii(d::Point<dim> q_point, std::array<d::Point<dim>,
    d::GeometryInfo<dim>::vertices_per_cell> & vertices){
  double rad = std::numeric_limits<double>::max();
  for(auto && v : vertices){
    double tmp = l_infty_distance(q_point, v);
    rad = tmp > rad ? rad : tmp;
  }
  return rad;
}

template<int dimension, int n_vertices>
d::Point<dimension> compute_barycenter(
    std::array<d::Point<dimension>, n_vertices> & vertices){

  d::Point<dimension> bc;
  for (unsigned int d=0; d < dimension; ++d) {
    bc[d] = 0;
  }

  for(auto && v : vertices){
    bc += v;
  }
  bc /= n_vertices;
  return bc;
}

template <int dimension>
class MeshGenerator {
public:
  d::Triangulation<dimension> triangulation;
  d::FE_Q<dimension>          fe;
  d::DoFHandler<dimension>    dof_handler;
  d::QGauss<dimension>              quadrature_formula;

  double box_a, box_b;

  // q: quad order
  // level: initial (uniform) level
  MeshGenerator(int q, int level):
      triangulation(d::Triangulation<dimension>::maximum_smoothing),
      fe(q),
      dof_handler(triangulation),
      quadrature_formula(q),
      box_a(a),
      box_b(b)
  {
    d::GridGenerator::hyper_cube(triangulation, box_a, box_b);
    if (level > 1) {
      triangulation.refine_global(level - 1);
    }
    this->dof_handler.distribute_dofs(fe);
  }

  // customized bounding box
  MeshGenerator(int q, int level, double aa, double bb):
      triangulation(d::Triangulation<dimension>::maximum_smoothing),
      fe(q),
      dof_handler(triangulation),
      quadrature_formula(q),
      box_a(aa),
      box_b(bb)
  {
    d::GridGenerator::hyper_cube(triangulation, box_a, box_b);
    if (level > 1) {
      triangulation.refine_global(level - 1);
    }
    this->dof_handler.distribute_dofs(fe);
  }

  // NOTE: Copy constructor is not supported since dealii::DoFHandler is not
  // copy constructable, but a dummy copy constructor is required by
  // boost.python to build the code.
  MeshGenerator(const MeshGenerator<dimension>&):
    MeshGenerator(1, 1){};

  std::string greet() { return "Hello from MeshGen."; }

  void generate_gmsh(p::object &fn) {
    std::string filename = p::extract<std::string>(fn);
    std::ofstream output_file(filename);
    d::GridOut().write_msh(this->triangulation, output_file);
  }

  np::ndarray get_q_points() {
    np::dtype dtype = np::dtype::get_builtin<double>();

    const unsigned int n_q_points = this->quadrature_formula.size();
    const int total_n_q_points = triangulation.n_active_cells() * n_q_points;
    Py_intptr_t pt_shape[2] = {total_n_q_points, dimension};

    d::FEValues<dimension>   fe_values(this->fe, this->quadrature_formula,
        d::update_quadrature_points);
    np::ndarray points = np::zeros(2, pt_shape, dtype);
    auto ptp           = reinterpret_cast<double*>(points.get_data());

    auto cell = dof_handler.begin_active();
    auto endc = dof_handler.end();
    for (; cell != endc; ++cell) {
      fe_values.reinit(cell);
      std::vector<d::Point<dimension>> q_points = fe_values.get_quadrature_points();

      for (auto&& point : q_points) {
        for (unsigned int d = 0; d < dimension; ++d) {
          // For a preferred ordering of quad points
          // This does not change anything but the ordering due to the symmetry
          // in each direction (as long as the subdivided boxes are cubes), not
          // even weights are changed.
          *ptp = point[dim-1-d];
          ++ptp;
        }
      }
    }

    return points;
  }

  np::ndarray get_q_weights(){
    np::dtype dtype = np::dtype::get_builtin<double>();

    const unsigned int n_q_points = this->quadrature_formula.size();
    const int total_n_q_points = this->triangulation.n_active_cells() * n_q_points;
    Py_intptr_t w_shape[1]       = {total_n_q_points};

    d::FEValues<dimension>   fe_values(this->fe, this->quadrature_formula,
        d::update_JxW_values);
    np::ndarray weights = np::zeros(1, w_shape, dtype);
    auto ptw = reinterpret_cast<double*>(weights.get_data());

    auto cell = this->dof_handler.begin_active();
    auto endc = this->dof_handler.end();
    for (; cell != endc; ++cell) {
      fe_values.reinit(cell);
      std::vector<double> q_weights;

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
        q_weights.push_back(fe_values.JxW(q_index));
      }

      for (auto&& weight : q_weights) {
          *ptw = weight;
          ++ptw;
      }
    }

    return weights;
  }

  np::ndarray get_cell_measures(){
    np::dtype dtype = np::dtype::get_builtin<double>();
    Py_intptr_t m_shape[1] = {this->triangulation.n_active_cells()};
    np::ndarray measures = np::zeros(1, m_shape, dtype);
    auto ptm = reinterpret_cast<double*>(measures.get_data());

    auto cell = this->dof_handler.begin_active();
    auto endc = this->dof_handler.end();
    for (; cell != endc; ++cell) {
      *ptm = cell->measure();
      ++ptm;
    }

    return measures;
  }

  np::ndarray get_cell_centers(){
    np::dtype dtype = np::dtype::get_builtin<double>();
    Py_intptr_t c_shape[2] = {this->triangulation.n_active_cells(), dimension};
    np::ndarray centers = np::zeros(2, c_shape, dtype);
    auto ptc = reinterpret_cast<double*>(centers.get_data());

    auto cell = this->dof_handler.begin_active();
    auto endc = this->dof_handler.end();
    for (; cell != endc; ++cell) {
      std::array<d::Point<dimension>,d::GeometryInfo<dimension>::vertices_per_cell> vertices;
      for (unsigned int v=0; v<d::GeometryInfo<dimension>::vertices_per_cell; ++v) {
        vertices[v] = cell->vertex(v);
      }
      auto barycenter = compute_barycenter<dimension,
                          d::GeometryInfo<dimension>::vertices_per_cell>(vertices);
      for (unsigned int d = 0; d < dimension; ++d) {
        *ptc = barycenter[dimension-1-d];
        ++ptc;
      }
    }

    return centers;
  }

  // Interface for dealii::GridRefinement::refine_and_coarsen_fixed_number
  void refine_and_coarsen_fixed_number(double* criteria,
                                       const double top_fraction_of_cells,
                                       const double bottom_fraction_of_cells) {
    d::Vector<double> ctr;
    ctr.reinit(this->triangulation.n_active_cells());
    double* ptc = criteria;
    for (unsigned int c = 0; c < this->triangulation.n_active_cells(); ++c){
      ctr[c] = *ptc;
      ++ptc;
    }

    d::GridRefinement::refine_and_coarsen_fixed_number (this->triangulation,
                                                        ctr,
                                                        top_fraction_of_cells,
                                                        bottom_fraction_of_cells);
  }

  int n_active_cells() {
    return this->triangulation.n_active_cells();
  }

  // Do both preparation for refinement and coarsening as well as mesh smoothing.
  // The function returns whether some cells' flagging has been changed in the process.
  bool prepare_coarsening_and_refinement() {
    return this->triangulation.prepare_coarsening_and_refinement();
  }

  void execute_coarsening_and_refinement() {
    this->triangulation.execute_coarsening_and_refinement();
    this->dof_handler.distribute_dofs(fe);
  }

  // Driver function for mesh adaptivity
  void update_mesh(np::ndarray const & criteria,
                   const double top_fraction_of_cells,
                   const double bottom_fraction_of_cells) {
    if (criteria.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type in the criteria");
        p::throw_error_already_set();
    }
    if (criteria.get_nd() != 1) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions of the criteria");
        p::throw_error_already_set();
    }

    // Refining top 1/3 in 2D doubles the number of cells
    double * iter = reinterpret_cast<double*>(criteria.get_data());
    this->refine_and_coarsen_fixed_number(iter, top_fraction_of_cells,
                                          bottom_fraction_of_cells);

    this->prepare_coarsening_and_refinement();
    this->execute_coarsening_and_refinement();
  }

  // Show some info about the mesh
  void print_info() {
    std::cout << "Number of active cells: " << this->triangulation.n_active_cells()
              << std::endl;

    d::FEValues<dimension>   fe_values(fe, quadrature_formula, d::update_quadrature_points);
    const unsigned int n_q_points = quadrature_formula.size();
    std::cout << "Number of quad points per cell: " << n_q_points << std::endl;
  }

};
typedef MeshGenerator<2> MeshGen2D;
typedef MeshGenerator<3> MeshGen3D;


// A legacy one-stop mesh generation function, returns quad points, weights
// and some spacing info within a tuple.
template <int dim>
p::tuple make_uniform_cubic_grid_details(int q, int level) {
  np::dtype dtype = np::dtype::get_builtin<double>();

  d::Triangulation<dim> triangulation;
  d::FE_Q<dim>          fe(q);
  d::DoFHandler<dim>    dof_handler(triangulation);
  d::QGauss<dim>        quadrature_formula(q);

  d::GridGenerator::hyper_cube(triangulation, a, b);
  if (level > 1) {
    triangulation.refine_global(level - 1);
  }
  dof_handler.distribute_dofs(fe);
  //std::cout << "Number of active cells: " << triangulation.n_active_cells()
            //<< std::endl;

  d::FEValues<dim>   fe_values(fe, quadrature_formula, update_flags);
  const unsigned int n_q_points = quadrature_formula.size();
  //std::cout << "Number of quad points per cell: " << n_q_points << std::endl;

  const int   total_n_q_points = triangulation.n_active_cells() * n_q_points;
  Py_intptr_t pt_shape[2]      = {total_n_q_points, dim};
  Py_intptr_t w_shape[1]       = {total_n_q_points};
  // Quad points
  np::ndarray points           = np::zeros(2, pt_shape, dtype);
  // Quad weights
  np::ndarray weights          = np::zeros(1, w_shape, dtype);
  // Distance (l_infty) to the closest cell vertex
  // (used for reconstructing the mesh in boxtree)
  np::ndarray radii            = np::zeros(1, w_shape, dtype);
  auto        ptp              = reinterpret_cast<double*>(points.get_data());
  auto        ptw              = reinterpret_cast<double*>(weights.get_data());
  auto        ptr              = reinterpret_cast<double*>(radii.get_data());

  // For margins
  std::array<double,dim*2> margins;
  std::array<double*, dim*2> mpts;
  margins.fill(std::numeric_limits<double>::max());

  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
    fe_values.reinit(cell);
    std::vector<d::Point<dim>> q_points = fe_values.get_quadrature_points();
    std::vector<double>        q_weights;
    std::array<d::Point<dim>,d::GeometryInfo<dim>::vertices_per_cell> vertices;

    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
      q_weights.push_back(fe_values.JxW(q_index));
    }

    for (unsigned int v=0; v<d::GeometryInfo<dim>::vertices_per_cell; ++v) {
      vertices[v] = cell->vertex(v);
    }

    for (auto&& point : q_points) {
      for (unsigned int d = 0; d < dim; ++d) {
        // For a preferred ordering of quad points
        // This does not change anything but the ordering due to the symmetry
        // in each direction (as long as the subdivided boxes are cubes), not
        // even weights are changed.
        *ptp = point[dim-1-d];
        ++ptp;
      }
      if( std::abs(point[0] - a) < margins[0] ){
        margins[0] = std::abs(point[0] - a);
        mpts[0] = ptr;
      }
      if( std::abs(point[0] - b) < margins[1] ){
        margins[1] = std::abs(point[0] - b);
        mpts[1] = ptr;
      }
      if( std::abs(point[1] - a) < margins[2] ){
        margins[2] = std::abs(point[1] - a);
        mpts[2] = ptr;
      }
      if( std::abs(point[1] - b) < margins[3] ){
        margins[3] = std::abs(point[1] - b);
        mpts[3] = ptr;
      }
      ++ptr;
    }

    for (auto&& weight : q_weights) {
      *ptw = weight;
      ++ptw;
    }
  }

  for (unsigned int i=0; i < dim*2; ++i) {
    *(mpts[i]) = margins[i];
  }

  p::tuple result = p::make_tuple(points, weights, radii);
  return result;
}

p::tuple make_uniform_cubic_grid(int q, int level, int dim) {
  if (dim == 1) {
    return make_uniform_cubic_grid_details<1>(q, level);
  }
  else if (dim == 2) {
    return make_uniform_cubic_grid_details<2>(q, level);
  }
  else if (dim == 3) {
    return make_uniform_cubic_grid_details<3>(q, level);
  }
  else {
    std::cout << "Dimension must be 1,2 or 3." << std::endl;
    return make_uniform_cubic_grid_details<3>(q, level);
  }
}

BOOST_PYTHON_MODULE(meshgen_dealii) {
  using namespace boost::python;
  Py_Initialize();
  np::initialize();

  docstring_options doc_options(/*show_all=*/true);

  def("greet", greet,
      "Greentings! This module handles generation of quadrature points & "
      "weights.\n\nThe points and weights follow Legendre-Gauss quadrature "
      "rules.");

  def("make_uniform_cubic_grid", make_uniform_cubic_grid,
      (p::arg("degree") = 3, p::arg("level") = 2, p::arg("dim") = 2), "Make a simple cubic grid.");

  class_<MeshGen2D>("MeshGen2D", init<int, int>())
        .def(init<int, int, double, double>())
        .def("greet", &MeshGen2D::greet)
        .def("get_q_points", &MeshGen2D::get_q_points)
        .def("get_q_weights", &MeshGen2D::get_q_weights)
        .def("get_cell_centers", &MeshGen2D::get_cell_centers)
        .def("get_cell_measures", &MeshGen2D::get_cell_measures)
        .def("n_active_cells", &MeshGen2D::n_active_cells)
        .def("prepare_coarsening_and_refinement", &MeshGen2D::prepare_coarsening_and_refinement)
        .def("execute_coarsening_and_refinement", &MeshGen2D::execute_coarsening_and_refinement)
        .def("update_mesh", &MeshGen2D::update_mesh)
        .def("print_info", &MeshGen2D::print_info)
        .def("generate_gmsh", &MeshGen2D::generate_gmsh)
    ;

  class_<MeshGen3D>("MeshGen3D", init<int, int>())
        .def(init<int, int, double, double>())
        .def("greet", &MeshGen3D::greet)
        .def("get_q_points", &MeshGen3D::get_q_points)
        .def("get_q_weights", &MeshGen3D::get_q_weights)
        .def("get_cell_centers", &MeshGen3D::get_cell_centers)
        .def("get_cell_measures", &MeshGen3D::get_cell_measures)
        .def("n_active_cells", &MeshGen3D::n_active_cells)
        .def("prepare_coarsening_and_refinement", &MeshGen3D::prepare_coarsening_and_refinement)
        .def("execute_coarsening_and_refinement", &MeshGen3D::execute_coarsening_and_refinement)
        .def("update_mesh", &MeshGen3D::update_mesh)
        .def("print_info", &MeshGen3D::print_info)
        .def("generate_gmsh", &MeshGen3D::generate_gmsh)
    ;
}
