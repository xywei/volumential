#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <array>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

namespace py = pybind11;
namespace d  = dealii;

constexpr int        dim = 2;
constexpr double     a   = -1;
constexpr double     b   = 1;
const d::UpdateFlags update_flags =
    d::update_quadrature_points | d::update_JxW_values;

void greet() { std::cout << "Hello from meshgen11_dealii." << std::endl; }

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
  d::QGauss<dim>              quadrature_formula;

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

  void generate_gmsh(const std::string fn) {
    std::string filename = fn;
    std::ofstream output_file(filename);
    d::GridOut().write_msh(this->triangulation, output_file);
  }

  py::array get_q_points() {
    py::dtype dtype = py::dtype::of<double>();

    const size_t n_q_points = this->quadrature_formula.size();
    const size_t total_n_q_points = triangulation.n_active_cells() * n_q_points;

    std::array<size_t, 2> pt_shape = {total_n_q_points, dimension};
    std::array<size_t, 2> pt_strides = {dimension*sizeof(double), sizeof(double)};

    d::FEValues<dimension>   fe_values(this->fe, this->quadrature_formula,
        d::update_quadrature_points);
    std::vector<double> points(pt_shape[0] * pt_shape[1], 0);
    auto ptp = points.data();

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

    return py::array(py::buffer_info(points.data(), sizeof(double),
          py::format_descriptor<double>::value,
          2, pt_shape, pt_strides));
  }

  py::array get_q_weights(){
    py::dtype dtype = py::dtype::of<double>();

    const size_t n_q_points = this->quadrature_formula.size();
    const size_t total_n_q_points = this->triangulation.n_active_cells() * n_q_points;

    std::array<size_t, 1> w_shape = {total_n_q_points};
    std::array<size_t, 1> w_strides = {sizeof(double)};

    d::FEValues<dimension>   fe_values(this->fe, this->quadrature_formula,
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

      for (auto&& weight : q_weights) {
          *ptw = weight;
          ++ptw;
      }
    }

    return py::array(py::buffer_info(weights.data(), sizeof(double),
          py::format_descriptor<double>::value,
          1, w_shape, w_strides));
  }

  py::array get_cell_measures(){
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
          py::format_descriptor<double>::value,
          1, m_shape, m_strides));
  }

  py::array get_cell_centers(){
    py::dtype dtype = py::dtype::of<double>();
    std::array<size_t, 2> c_shape = {this->triangulation.n_active_cells(), dimension};
    std::array<size_t, 2> c_strides = {dimension*sizeof(double), sizeof(double)};
    std::vector<double> centers (c_shape[0]*c_shape[1], 0);
    auto ptc = centers.data();

    auto cell = this->dof_handler.begin_active();
    auto endc = this->dof_handler.end();
    for (; cell != endc; ++cell) {
      std::array<d::Point<dim>,d::GeometryInfo<dim>::vertices_per_cell> vertices;
      for (unsigned int v=0; v<d::GeometryInfo<dim>::vertices_per_cell; ++v) {
        vertices[v] = cell->vertex(v);
      }
      auto barycenter = compute_barycenter<dimension,
                          d::GeometryInfo<dim>::vertices_per_cell>(vertices);
      for (unsigned int d = 0; d < dimension; ++d) {
        *ptc = barycenter[dimension-1-d];
        ++ptc;
      }
    }

    return py::array(py::buffer_info(centers.data(), sizeof(double),
            py::format_descriptor<double>::value,
            2, c_shape, c_strides));
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
  void update_mesh(py::array_t<double> & criteria,
                   const double top_fraction_of_cells,
                   const double bottom_fraction_of_cells) {
    py::buffer_info info_crit = criteria.request();
    if (info_crit.ndim != 1) {
      throw std::runtime_error("Number of dimensions must be one");
    }

    // Refining top 1/3 in 2D doubles the number of cells
    double * iter = (double*) info_crit.ptr;
    this->refine_and_coarsen_fixed_number(iter, top_fraction_of_cells,
                                          bottom_fraction_of_cells);

    this->prepare_coarsening_and_refinement();
    this->execute_coarsening_and_refinement();
  }

  // Show some info about the mesh
  void print_info() {
    std::cout << "Number of active cells: " << this->triangulation.n_active_cells()
              << std::endl;

    d::FEValues<dim>   fe_values(fe, quadrature_formula, d::update_quadrature_points);
    const unsigned int n_q_points = quadrature_formula.size();
    std::cout << "Number of quad points per cell: " << n_q_points << std::endl;
  }

};
typedef MeshGenerator<2> MeshGen2D;


// A legacy one-stop mesh generation function, returns quad points, weights
// and some spacing info within a tuple.
py::tuple make_uniform_cubic_grid(int q, int level) {
  py::dtype dtype = py::dtype::of<double>();

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
  const size_t n_q_points = quadrature_formula.size();
  //std::cout << "Number of quad points per cell: " << n_q_points << std::endl;

  const size_t total_n_q_points = triangulation.n_active_cells() * n_q_points;

  std::array<size_t, 2> pt_shape   = {total_n_q_points, dim};
  std::array<size_t, 2> pt_strides = {dim*sizeof(double), sizeof(double)};
  std::array<size_t, 1> w_shape    = {total_n_q_points};
  std::array<size_t, 1> w_strides  = {sizeof(double)};

  // Quad points
  std::vector<double> points(pt_shape[0] * pt_shape[1], 0);
  // Quad weights
  std::vector<double> weights(w_shape[0], 0);
  // Distance (l_infty) to the closest cell vertex
  // (used for reconstructing the mesh in boxtree)
  std::vector<double> radii(w_shape[0], 0);
  auto        ptp   = points.data();
  auto        ptw   = weights.data();
  auto        ptr   = radii.data();

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

  py::tuple result = py::make_tuple(
      py::array(py::buffer_info(points.data(), sizeof(double),
          py::format_descriptor<double>::value,
          2, pt_shape, pt_strides)),
      py::array(py::buffer_info(weights.data(), sizeof(double),
          py::format_descriptor<double>::value,
          1, w_shape, w_strides)),
      py::array(py::buffer_info(radii.data(), sizeof(double),
          py::format_descriptor<double>::value,
          1, w_shape, w_strides)));

  return result;
}

PYBIND11_MODULE(meshgen, m) {
    m.doc() = "A mesh generator for volumential.";

    m.def("greet", &greet, "Greetings! This is meshgen11.");

    m.def("make_uniform_cubic_grid", &make_uniform_cubic_grid,
        "Make a simple grid", py::arg("degree"), py::arg("level"));

    py::class_<MeshGen2D>(m, "MeshGen2D")
      .def(py::init<int, int>())
      .def(py::init<int, int, double, double>())
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
      .def("generate_gmsh", &MeshGen2D::generate_gmsh);
}
