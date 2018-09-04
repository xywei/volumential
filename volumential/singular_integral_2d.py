from __future__ import division

import logging

import numpy as np
import scipy.integrate.quadrature as quad
"""
:mod:
    The 2D singular integrals are computed using the transform described in
    http://link.springer.com/10.1007/BF00370482.
"""

logger = logging.getLogger(__name__)

# {{{ quadrature on rectangular box with no singularity

quad_points_x = np.array([])
quad_points_y = np.array([])
quad_weights = np.array([])


def update_qquad_leggauss_formula(
        deg1, deg2):

    x1, w1 = np.polynomial.legendre.leggauss(
        deg1)
    x1 = (x1 + 1) / 2
    w1 = w1 / 2

    x2, w2 = np.polynomial.legendre.leggauss(
        deg2)
    x2 = (x2 + 1) / 2
    w2 = w2 / 2

    quad_points_x, quad_points_y = np.meshgrid(
        x1, x2)
    ww1, ww2 = np.meshgrid(w1, w2)
    global quad_weights
    quad_weights = ww1 * ww2


def qquad(func,
          a,
          b,
          c,
          d,
          args=(),
          tol=1.49e-08,
          rtol=1.49e-08,
          maxitero=50,
          maxiteri=50,
          vec_func=False,
          minitero=1,
          miniteri=1,
          method="Adaptive"):
    """Computes a (tensor product) double integral.

    Integrate func on [a, b]X[c, d] using Gaussian quadrature with absolute
    tolerance tol.

    :param func: A double variable Python function or method to integrate.
    :type func: function.
    :param a: Lower-left corner of integration region.
    :type a: float.
    :param b: Lower-right corner of integration region.
    :type b: float.
    :param c: Upper-left corner of integration region.
    :type c: float.
    :param d: Upper-right corner of integration region.
    :type d: float.
    :param args: Extra arguments to pass to function.
    :type args: tuple, optional.
    :param tol: rtol Iteration stops when error between last two iterates is
                less than tol OR the relative change is less than rtol.
    :type tol: float, optional.
    :param rtol: Iteration stops when error between last two iterates is less
                than tol OR the relative change is less than rtol.
    :type rtol: float, optional.
    :param maxitero: Maximum order of outer Gaussian quadrature.
    :type maxitero: int, optional.
    :param maxiteri: Maximum order of inner Gaussian quadrature.
    :type maxiteri: int, optional.
    :param vec_func: True if func handles arrays as arguments (is a “vector”
                function). Default is True.
    :type vec_func: bool, optional.
    :param minitero: Minimum order of outer Gaussian quadrature.
    :type minitero: int, optional.
    :param miniteri: Minimum order of inner Gaussian quadrature.
    :type miniteri: int, optional.

    :returns:
        - **val**: Gaussian quadrature approximation (within tolerance) to integral.
        - **err**: Difference between last two estimates of the integral.
    :rtype: tuple(float,float).
    """

    l1 = b - a
    l2 = d - c
    assert (l1 > 0)
    assert (l2 > 0)
    toli = tol / l2
    rtoli = rtol / l2

    if method == "Adaptive":

        # Using lambda for readability
        def outer_integrand(y): return quad(  # NOQA
            lambda x: func(x, y, *args), a, b, (), toli, rtoli, maxiteri, vec_func, miniteri)[0]  # NOQA

        # Is there a simple way to retrieve err info from the inner quad calls?

        val, err = quad(
            outer_integrand, c, d, (),
            tol, rtol, maxitero,
            vec_func, minitero)

    elif method == "LegGauss":

        # Legendre-Gauss quadrature with orders equal to maxiters

        val = 0
        err = 0

    else:

        raise NotImplementedError(
            "Unsupported quad method: "
            + method)

    return (val, err)


# }}}

# {{{ affine mappings


def solve_affine_map_2d(source_tria,
                        target_tria):
    """Computes the affine map and its inverse that maps the source_tria to
       target_tria.

       :param source_tria: The triangle to be mapped.
       :type source_tria:
            tuple(tuple(float,float),tuple(float,float),tuple(float,float)).
       :param target_tria: The triangle to map to.
       :type target_tria:
            tuple(tuple(float,float),tuple(float,float),tuple(float,float)).

       :returns:
        - **mapping**: the forward map.
        - **J**: the Jacobian.
        - **invmap**: the inverse map.
        - **invJ**: the Jacobian of inverse map.
       :rtype:
        tuple(lambda, float, lambda, float)
    """
    assert (len(source_tria) == 3)
    for p in source_tria:
        assert (len(p) == 2)

    assert (len(target_tria) == 3)
    for p in target_tria:
        assert (len(p) == 2)

    # DOFs: A11, A12, A21, A22, b1, b2
    rhs = np.array([
        target_tria[0][0],
        target_tria[0][1],
        target_tria[1][0],
        target_tria[1][1],
        target_tria[2][0],
        target_tria[2][1]
    ])

    coef = np.array([
        [
            source_tria[0][0],
            source_tria[0][1], 0, 0, 1,
            0
        ],
        [
            0, 0, source_tria[0][0],
            source_tria[0][1], 0, 1
        ],
        [
            source_tria[1][0],
            source_tria[1][1], 0, 0, 1,
            0
        ],
        [
            0, 0, source_tria[1][0],
            source_tria[1][1], 0, 1
        ],
        [
            source_tria[2][0],
            source_tria[2][1], 0, 0, 1,
            0
        ],
        [
            0, 0, source_tria[2][0],
            source_tria[2][1], 0, 1
        ],
    ])

    # x, residuals, _, _ = np.linalg.lstsq(coef, rhs)
    # assert (np.allclose(residuals, 0))
    try:
        x = np.linalg.solve(coef, rhs)
    except np.linalg.linalg.LinAlgError:
        print('')
        print("source:", source_tria)
        print("target:", target_tria)
        raise SystemExit(
            "Error: Singular source triangle encountered"
        )
    assert (len(x) == 6)
    assert (np.allclose(
        np.dot(coef, x), rhs))

    a = np.array([[x[0], x[1]],
                  [x[2], x[3]]])
    b = np.array([x[4], x[5]])

    # Using default value is the idiomatic way to "capture by value"
    mapping = lambda x, a=a, b=b: a.dot(np.array(x)) + b  # NOQA
    jacob = np.linalg.det(a)

    inva = np.linalg.inv(a)
    invb = -inva.dot(b)
    invmap = lambda x, a=inva, b=invb: inva.dot(np.array(x)) + invb  # NOQA
    inv_jacob = np.linalg.det(inva)

    assert (
        np.abs(jacob * inv_jacob - 1) <
        1e-12)

    return (mapping, jacob, invmap,
            inv_jacob)


# }}}

# {{{ standard-triangle-to-rectangle mappings


def tria2rect_map_2d():
    """Returns the mapping and its inverse that maps a template triangle to a
       template rectangle.

            - Template triangle [T]: (0,0)--(1,0)--(0,1)--(0,0)
            - Template rectangle [R]: (0,0)--(1,0)--(1,pi/2)--(0,pi/2)--(0,0)

       :returns: The mapping, its Jacobian, its inverse, and the Jacobian of its
                inverse. Note that the Jacobians are returned as lambdas since
                they are not constants.
       :rtype: tuple(lambda, lambda, lambda, lambda)
    """

    # (x,y) --> (rho, theta): T --> R
    def mapping(x):
        return (x[0] + x[1], np.arctan2(
            np.sqrt(x[1]),
            np.sqrt(x[0])))

    def jacob(x):
        return 1 / (
            2 * np.sqrt(x[0] * x[1]))

    # x = rho * cos^2(theta), y = rho * sin^2(theta)
    # J = rho * sin(2*theta)
    def invmap(u):
        return (u[0] *
                (np.cos(u[1])**2),
                u[0] *
                (np.sin(u[1])**2))

    def inv_jacob(u):
        return u[0] * np.sin(2 * u[1])

    return (mapping, jacob, invmap,
            inv_jacob)


def is_in_t(pt):
    """Checks if a point is in the template triangle T.

       :param pt: The point to be checked.
       :type pt: tuple(float,float).

       :returns: True if pt is in T.
       :rtype: bool.
    """
    flag = True
    if pt[0] < 0 or pt[1] < 0:
        flag = False
    if pt[0] + pt[1] > 1:
        flag = False
    return flag


def is_in_r(pt,
            a=0,
            b=1,
            c=0,
            d=np.pi / 2):
    """Checks if a point is in the (template) rectangle R.

       :param pt: The point to be checked.
       :type pt: tuple(float,float).

       :returns: True if pt is in R.
       :rtype: bool.
    """
    flag = True
    if pt[0] < a or pt[1] < c:
        flag = False
    if pt[0] > b or pt[1] > d:
        flag = False
    return flag


# }}}

# {{{ quadrature on arbitrary triangle


def is_collinear(p0, p1, p2):
    # v1 = p0 --> p1
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    # v2 = p0 --> p2
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    # v1 cross v2 == 0 <==> collinearity
    return np.abs(x1 * y2 -
                  x2 * y1) < 1e-16


def is_positive_triangle(tria):
    p0 = tria[0]
    p1 = tria[1]
    p2 = tria[2]
    # v1 = p0 --> p1
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    # v2 = p0 --> p2
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    # v1 cross v2 > 0 <==> is positive
    return (x1 * y2 - x2 * y1) > 0


def tria_quad(func,
              tria,
              args=(),
              tol=1.49e-08,
              rtol=1.49e-08,
              maxiter=50,
              vec_func=True,
              miniter=1):
    """Computes a double integral on a general triangular region.

    Integrate func on tria by transforming the region into a rectangle and
    using Gaussian quadrature with absolute tolerance tol.

    The integrand, func, is allowed to have singularity at most $O(r)$ at the
    first virtex of the tiangle. It is okay if func does not evaluate at the
    singular point. This function handles that automatically.

    :param func: A double variable Python function or method to integrate.
    :type func: function.
    :param tria: The triangular region to do quadrature.
    :type tria:
        tuple(tuple(float,float), tuple(float,float), tuple(float,float)).
    :param args: Extra arguments to pass to function.
    :type args: tuple, optional.
    :param tol: rtol Iteration stops when error between last two iterates is
        less than tol OR the relative change is less than rtol.
    :type tol: float, optional.
    :param rtol: Iteration stops when error between last two iterates is less
        than tol OR the relative change is less than rtol.
    :type rtol: float, optional.
    :param maxiter: Maximum order of Gaussian quadrature.
    :type maxiter: int, optional.
    :param vec_func: True if func handles arrays as arguments
        (is a “vector” function). Default is True.
    :type vec_func: bool, optional.
    :param miniter: Minimum order of Gaussian quadrature.
    :type miniter: int, optional.

    :returns:
        - **val**: Gaussian quadrature approximation (within tolerance)
            to integral.
        - **err**: Difference between last two estimates of the integral.
    :rtype: tuple(float,float).
    """

    assert (len(tria) == 3)
    for p in tria:
        assert (len(p) == 2)

    # Handle degenerate triangles
    if is_collinear(*tria):
        return (0., 0.)

    # The function must be regular at the last two vertices
    assert (np.isfinite(
        func(*tria[1], *args)))
    assert (np.isfinite(
        func(*tria[2], *args)))

    # Solve for transforms
    template_tria = ((0, 0), (1, 0),
                     (0, 1))
    afmp, j_afmp, inv_afmp, j_inv_afmp = solve_affine_map_2d(
        tria, template_tria)
    nlmp, j_nlmp, inv_nlmp, j_inv_nlmp = tria2rect_map_2d(
    )

    # tria --> rect
    def mapping(x, y):
        return (nlmp(afmp((x, y))))

    def jacobian(x, y):
        return (j_afmp *
                j_nlmp(afmp((x, y))))

    # rect --> tria
    def inv_mapping(rho, theta):
        return (inv_afmp(
            inv_nlmp((rho, theta))))

    def inv_jacobian(rho, theta):
        return (j_inv_afmp * j_inv_nlmp(
            (rho, theta)))

    # Transformed function is defined on [0,1]X[0,pi/2]
    def transformed_func(rho, theta):
        preimage = inv_mapping(
            rho, theta)
        return func(*preimage, *args)

    # Transformed function, when multiplied by jacobian, should have no
    # singularity (numerically special treatment still needed)

    # integrand = func * jacobian

    def integrand(rho, theta):
        prior = transformed_func(
            rho, theta) * inv_jacobian(
                rho, theta)
        # If something blows up, it is near the singular point
        if (~np.isfinite(prior)):
            assert (rho < 1e-3)
            assert (inv_jacobian(rho, theta) < 1e-6)
            prior = 0
        return prior

    return qquad(
        func=integrand,
        a=0,
        b=1,
        c=0,
        d=np.pi / 2,
        args=(),
        tol=tol,
        rtol=rtol,
        maxiteri=maxiter,
        maxitero=maxiter,
        vec_func=False,
        miniteri=miniter,
        minitero=miniter)


# }}}

# {{{ quadrature on a 2d box with a singular point inside


def box_quad(func,
             a,
             b,
             c,
             d,
             singular_point,
             args=(),
             tol=1.49e-08,
             rtol=1.49e-08,
             maxiter=50,
             vec_func=True,
             miniter=1):
    """Computes a (tensor product) double integral, with the integrand being
        singular at some point inside the region.

    Integrate func on [a, b]X[c, d] using transformed Gaussian quadrature with
    absolute tolerance tol.

    :param func: A double variable Python function or method to integrate.
    :type func: function.
    :param a: Lower-left corner of integration region.
    :type a: float.
    :param b: Lower-right corner of integration region.
    :type b: float.
    :param c: Upper-left corner of integration region.
    :type c: float.
    :param d: Upper-right corner of integration region.
    :type d: float.
    :param singular_point: The singular point of the integrand func.
    :type singular_point: tuple(float, float).
    :param args: Extra arguments to pass to function.
    :type args: tuple, optional.
    :param tol: rtol Iteration stops when error between last two iterates is
                less than tol OR the relative change is less than rtol.
    :type tol: float, optional.
    :param rtol: Iteration stops when error between last two iterates is less
                than tol OR the relative change is less than rtol.
    :type rtol: float, optional.
    :param maxiter: Maximum order of Gaussian quadrature.
    :type maxiter: int, optional.
    :param vec_func: True if func handles arrays as arguments (is a “vector”
                function). Default is True.
    :type vec_func: bool, optional.
    :param miniter: Minimum order of Gaussian quadrature.
    :type miniter: int, optional.

    :returns:
        - **val**: Gaussian quadrature approximation (within tolerance) to integral.
        - **err**: Difference between last two estimates of the integral.
    :rtype: tuple(float,float).
    """
    box = ((a, c), (b, c), (b, d), (a,
                                    d))

    if not isinstance(singular_point,
                      tuple):
        singular_point = (
            singular_point[0],
            singular_point[1])

    # When singular point is outside, project it onto the box bounday
    # This can import speed by not integrating around the actual singularity
    # when not necessary. (The splitting is still needed since it can be quite
    # close to singular).
    singular_point = (max(
        singular_point[0], a), max(
            singular_point[1], c))
    singular_point = (min(
        singular_point[0], b), min(
            singular_point[1], d))

    return quadri_quad(
        func, box, singular_point, args,
        tol, rtol, maxiter, vec_func,
        miniter)


# quadrilateral
def quadri_quad(func,
                quadrilateral,
                singular_point,
                args=(),
                tol=1.49e-08,
                rtol=1.49e-08,
                maxiter=50,
                vec_func=True,
                miniter=1):
    """Computes a double integral over a (non-twisted) quadrilateral, with the
        integrand being singular at some point inside the region.

    Integrate func on [a, b]X[c, d] using transformed Gaussian quadrature with
    absolute tolerance tol.

    :param func: A double variable Python function or method to integrate.
    :type func: function.
    :param quadrilateral: The integration region.
    :type quadrilateral: tuple(tuple(float,float),
        tuple(float,float), tuple(float,float), tuple(float,float)).
    :param singular_point: The singular point of the integrand func.
    :type singular_point: tuple(float, float).
    :param args: Extra arguments to pass to function.
    :type args: tuple, optional.
    :param tol: rtol Iteration stops when error between last two iterates is
                less than tol OR the relative change is less than rtol.
    :type tol: float, optional.
    :param rtol: Iteration stops when error between last two iterates is less
                than tol OR the relative change is less than rtol.
    :type rtol: float, optional.
    :param maxiter: Maximum order of Gaussian quadrature.
    :type maxiter: int, optional.
    :param vec_func: True if func handles arrays as arguments (is a “vector”
                function). Default is True.
    :type vec_func: bool, optional.
    :param miniter: Minimum order of Gaussian quadrature.
    :type miniter: int, optional.

    :returns:
        - **val**: Gaussian quadrature approximation (within tolerance) to integral.
        - **err**: Difference between last two estimates of the integral.
    :rtype: tuple(float,float).
    """
    assert (len(quadrilateral) == 4)
    for p in quadrilateral:
        assert (len(p) == 2)

    # split the quadrilateral into four triangles
    trias = [
        (singular_point,
         quadrilateral[0],
         quadrilateral[1]),
        (singular_point,
         quadrilateral[1],
         quadrilateral[2]),
        (singular_point,
         quadrilateral[2],
         quadrilateral[3]),
        (singular_point,
         quadrilateral[3],
         quadrilateral[0]),
    ]

    for tria in trias:
        if not is_positive_triangle(tria):
            assert(is_collinear(*tria))

    val = np.zeros(4)
    err = np.zeros(4)

    for i in range(4):
        val[i], err[i] = tria_quad(
            func, trias[i], args, tol,
            rtol, maxiter, vec_func,
            miniter)

    integral = np.sum(val)
    error = np.linalg.norm(err)

    return (integral, error)


# }}}


'''
class DesingularizationMapping:
    def __init__(self, nquad_points_1d):

def build_singular_box_quadrature(
        kernel,
        desing_mapping,

        ):
    """
    :arg kernel: an instance of :class:`sumpy.kernel.Kernel`
    :arg desing_mapping: an instance of :class:`sumpy.kernel.Kernel`
    """
'''


# vim: filetype=pyopencl.python:fdm=marker
