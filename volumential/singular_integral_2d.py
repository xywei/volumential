__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
from numbers import Number

import mpmath
import numpy as np
import scipy as sp


__doc__ = """The 2D singular integrals are computed using the transform
described in http://link.springer.com/10.1007/BF00370482.

.. autofunction:: box_quad
"""

logger = logging.getLogger(__name__)

# {{{ quadrature on rectangular box with no singularity

quad_points_x = np.array([])
quad_points_y = np.array([])
quad_weights = np.array([])


def _to_float_or_array(value):
    if isinstance(value, Number):
        return float(value)

    arr = np.asarray(value)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0])

    return arr


def _meets_tolerance(current, previous, tol, rtol):
    delta = np.asarray(current) - np.asarray(previous)
    err = np.linalg.norm(np.ravel(delta), ord=np.inf)
    scale = np.linalg.norm(np.ravel(np.asarray(current)), ord=np.inf)
    return err <= tol or err <= rtol * scale, err


def adaptive_quadrature(
    func,
    a,
    b,
    args=(),
    tol=1.49e-08,
    rtol=1.49e-08,
    maxiter=50,
    vec_func=False,
    miniter=1,
):
    """Approximate the removed ``scipy.integrate.quadrature`` API.

    The legacy code relied on quadrature order refinement semantics from older
    SciPy. Recreate the essential behavior by increasing the fixed Gauss order
    until successive iterates stabilize.
    """

    if maxiter < 1:
        raise ValueError("maxiter must be positive")

    miniter = max(1, miniter)
    maxiter = max(miniter, maxiter)

    def wrapped_func(x, *fargs):
        x_arr = np.atleast_1d(np.asarray(x))

        if vec_func:
            result = func(x_arr, *fargs)
        else:
            result = np.array([func(xi, *fargs) for xi in x_arr])

        result = np.asarray(result)
        if np.ndim(x) == 0:
            return result.reshape(-1)[0]

        return result

    previous = None
    current = None
    err = np.inf
    had_comparison = False

    for order in range(miniter, maxiter + 1):
        current = _to_float_or_array(
            sp.integrate.fixed_quad(wrapped_func, a, b, args=args, n=order)[0]
        )

        if previous is not None:
            had_comparison = True
            converged, err = _meets_tolerance(current, previous, tol, rtol)
            if converged:
                return current, err

        previous = current

    if current is None:
        raise RuntimeError("adaptive quadrature did not execute any iterations")

    if not had_comparison:
        err = 0.0

    return current, err


quad = adaptive_quadrature


def _tensor_product_fixed_quad(
    func, a, b, c, d, args=(), order_o=1, order_i=1, vec_func=False
):
    x_nodes, x_weights = np.polynomial.legendre.leggauss(order_i)
    y_nodes, y_weights = np.polynomial.legendre.leggauss(order_o)

    x_mapped = 0.5 * (b - a) * x_nodes + 0.5 * (a + b)
    y_mapped = 0.5 * (d - c) * y_nodes + 0.5 * (c + d)

    if vec_func:
        x_grid, y_grid = np.meshgrid(x_mapped, y_mapped, indexing="xy")
        values = np.asarray(func(x_grid, y_grid, *args))
    else:
        values = np.empty((order_o, order_i), dtype=np.float64)
        for iy, y_val in enumerate(y_mapped):
            for ix, x_val in enumerate(x_mapped):
                values[iy, ix] = _to_float_or_array(func(x_val, y_val, *args))

    weights = np.outer(y_weights, x_weights)
    return 0.25 * (b - a) * (d - c) * np.sum(weights * values, axis=(-2, -1))


def update_qquad_leggauss_formula(deg1, deg2):

    x1, w1 = np.polynomial.legendre.leggauss(deg1)
    x1 = (x1 + 1) / 2
    w1 = w1 / 2

    x2, w2 = np.polynomial.legendre.leggauss(deg2)
    x2 = (x2 + 1) / 2
    w2 = w2 / 2

    quad_points_x, quad_points_y = np.meshgrid(x1, x2)
    ww1, ww2 = np.meshgrid(w1, w2)
    global quad_weights
    quad_weights = ww1 * ww2


def qquad(
    func,
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
    method="Adaptive",
):
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
    :param vec_func: True if func handles arrays as arguments (is a "vector"
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
    assert l1 > 0
    assert l2 > 0

    if method == "Adaptive":
        previous = None
        val = None
        err = np.inf
        had_comparison = False
        max_steps = max(maxiteri - miniteri, maxitero - minitero) + 1

        for istep in range(max_steps):
            order_i = min(maxiteri, miniteri + istep)
            order_o = min(maxitero, minitero + istep)
            val = _tensor_product_fixed_quad(
                func,
                a,
                b,
                c,
                d,
                args=args,
                order_o=order_o,
                order_i=order_i,
                vec_func=vec_func,
            )

            if previous is not None:
                had_comparison = True
                converged, err = _meets_tolerance(val, previous, tol, rtol)
                if converged:
                    break

            previous = val

        if not had_comparison:
            err = 0.0

    elif method == "Gauss":
        # Gauss quadrature with orders equal to maxiters

        # Using lambda for readability
        def outer_integrand(y):
            return sp.integrate.fixed_quad(
                np.vectorize(lambda x: func(x, y, *args)), a, b, (), maxiteri
            )[0]

        # Is there a simple way to retrieve err info from the inner quad calls?

        val, err = sp.integrate.fixed_quad(
            np.vectorize(outer_integrand), c, d, (), maxitero
        )

        assert err is None

    else:
        raise NotImplementedError("Unsupported quad method: " + method)

    return (val, err)


# }}}

# {{{ affine mappings


def solve_affine_map_2d(source_tria, target_tria):
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
    assert len(source_tria) == 3
    for p in source_tria:
        assert len(p) == 2

    assert len(target_tria) == 3
    for p in target_tria:
        assert len(p) == 2

    # DOFs: A11, A12, A21, A22, b1, b2
    rhs = np.array(
        [
            target_tria[0][0],
            target_tria[0][1],
            target_tria[1][0],
            target_tria[1][1],
            target_tria[2][0],
            target_tria[2][1],
        ]
    )

    coef = np.array(
        [
            [source_tria[0][0], source_tria[0][1], 0, 0, 1, 0],
            [0, 0, source_tria[0][0], source_tria[0][1], 0, 1],
            [source_tria[1][0], source_tria[1][1], 0, 0, 1, 0],
            [0, 0, source_tria[1][0], source_tria[1][1], 0, 1],
            [source_tria[2][0], source_tria[2][1], 0, 0, 1, 0],
            [0, 0, source_tria[2][0], source_tria[2][1], 0, 1],
        ]
    )

    # x, residuals, _, _ = np.linalg.lstsq(coef, rhs)
    # assert (np.allclose(residuals, 0))
    try:
        x = np.linalg.solve(coef, rhs)
    except np.linalg.linalg.LinAlgError:
        print("")
        print("source:", source_tria)
        print("target:", target_tria)
        raise SystemExit("Error: Singular source triangle encountered")
    assert len(x) == 6
    assert np.allclose(np.dot(coef, x), rhs)

    a = np.array([[x[0], x[1]], [x[2], x[3]]])
    b = np.array([x[4], x[5]])

    # Using default value is the idiomatic way to "capture by value"
    mapping = lambda x, a=a, b=b: a.dot(np.array(x)) + b  # noqa: E731
    jacob = np.linalg.det(a)

    inva = np.linalg.inv(a)
    invb = -inva.dot(b)
    invmap = lambda x, a=inva, b=invb: inva.dot(np.array(x)) + invb  # noqa: E731
    inv_jacob = np.linalg.det(inva)

    assert np.abs(jacob * inv_jacob - 1) < 1e-12

    return (mapping, jacob, invmap, inv_jacob)


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
        return (x[0] + x[1], np.arctan2(np.sqrt(x[1]), np.sqrt(x[0])))

    def jacob(x):
        return 1 / (2 * np.sqrt(x[0] * x[1]))

    # x = rho * cos^2(theta), y = rho * sin^2(theta)
    # J = rho * sin(2*theta)
    def invmap(u):
        return (u[0] * (np.cos(u[1]) ** 2), u[0] * (np.sin(u[1]) ** 2))

    def inv_jacob(u):
        return u[0] * np.sin(2 * u[1])

    return (mapping, jacob, invmap, inv_jacob)


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


def is_in_r(pt, a=0, b=1, c=0, d=np.pi / 2):
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
    return np.abs(x1 * y2 - x2 * y1) < 1e-16


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


def tria_quad(
    func,
    tria,
    args=(),
    tol=1.49e-08,
    rtol=1.49e-08,
    maxiter=50,
    vec_func=True,
    miniter=1,
):
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
        (is a "vector" function). Default is True.
    :type vec_func: bool, optional.
    :param miniter: Minimum order of Gaussian quadrature.
    :type miniter: int, optional.

    :returns:
        - **val**: Gaussian quadrature approximation (within tolerance)
            to integral.
        - **err**: Difference between last two estimates of the integral.
    :rtype: tuple(float,float).
    """

    assert len(tria) == 3
    for p in tria:
        assert len(p) == 2

    # Handle degenerate triangles
    if is_collinear(*tria):
        return (0.0, 0.0)

    # The function must be regular at the last two vertices
    assert np.isfinite(func(tria[1][0], tria[1][1], *args))
    assert np.isfinite(func(tria[2][0], tria[2][1], *args))

    # Solve for transforms
    template_tria = ((0, 0), (1, 0), (0, 1))
    afmp, j_afmp, inv_afmp, j_inv_afmp = solve_affine_map_2d(tria, template_tria)
    nlmp, j_nlmp, inv_nlmp, j_inv_nlmp = tria2rect_map_2d()

    # tria --> rect
    def mapping(x, y):
        return nlmp(afmp((x, y)))

    def jacobian(x, y):
        return j_afmp * j_nlmp(afmp((x, y)))

    # rect --> tria
    def inv_mapping(rho, theta):
        return inv_afmp(inv_nlmp((rho, theta)))

    def inv_jacobian(rho, theta):
        value = np.asarray(j_inv_afmp * j_inv_nlmp((rho, theta)))
        return float(value.reshape(-1)[0])

    # Transformed function is defined on [0,1]X[0,pi/2]
    def transformed_func(rho, theta):
        preimage = inv_mapping(rho, theta)
        value = np.asarray(func(preimage[0], preimage[1], *args))
        return float(value.reshape(-1)[0])

    # Transformed function, when multiplied by jacobian, should have no
    # singularity (numerically special treatment still needed)

    # integrand = func * jacobian

    def integrand(rho, theta):
        prior = transformed_func(rho, theta) * inv_jacobian(rho, theta)
        # If something blows up, it is near the singular point
        if ~np.isfinite(prior):
            assert rho < 1e-3
            assert inv_jacobian(rho, theta) < 1e-6
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
        minitero=miniter,
    )


def tria_quad_duffy_radial(
    func,
    tria,
    args=(),
    radial_rule="tanh-sinh",
    deg_theta=20,
    radial_quad_order=61,
    mp_dps=50,
):
    assert len(tria) == 3
    for p in tria:
        assert len(p) == 2

    if is_collinear(*tria):
        return (0.0, 0.0)

    assert np.isfinite(func(tria[1][0], tria[1][1], *args))
    assert np.isfinite(func(tria[2][0], tria[2][1], *args))

    template_tria = ((0, 0), (1, 0), (0, 1))
    afmp, j_afmp, inv_afmp, j_inv_afmp = solve_affine_map_2d(tria, template_tria)
    nlmp, j_nlmp, inv_nlmp, j_inv_nlmp = tria2rect_map_2d()

    def inv_mapping(rho, theta):
        return inv_afmp(inv_nlmp((rho, theta)))

    def inv_jacobian(rho, theta):
        return j_inv_afmp * j_inv_nlmp((rho, theta))

    def transformed_func(rho, theta):
        preimage = inv_mapping(rho, theta)
        return func(preimage[0], preimage[1], *args)

    def integrand(rho, theta):
        prior = np.asarray(transformed_func(rho, theta) * inv_jacobian(rho, theta))
        prior = float(prior.reshape(-1)[0])
        if not np.isfinite(prior):
            if rho < 1.0e-14:
                return 0.0
            raise FloatingPointError((rho, theta, prior))
        return prior

    th_nodes, th_weights = sp.special.p_roots(deg_theta)
    th_nodes = 0.25 * np.pi * (th_nodes + 1.0)
    th_weights = 0.25 * np.pi * th_weights

    total = 0.0
    if radial_rule == "tanh-sinh":
        old_dps = mpmath.mp.dps
        mpmath.mp.dps = mp_dps
        try:
            for theta, wt in zip(th_nodes, th_weights):
                radial_val = mpmath.quadts(
                    lambda rho: integrand(float(rho), theta), [0, 1]
                )
                total += wt * float(radial_val)
        finally:
            mpmath.mp.dps = old_dps
    elif radial_rule == "tanh-sinh-fast":
        n = max(3, int(radial_quad_order))
        h = 1.0 / np.sqrt(n)
        k = np.arange(-n, n + 1, dtype=np.float64)
        t = h * k
        sh_t = np.sinh(t)
        ch_t = np.cosh(t)
        arg = 0.5 * np.pi * sh_t
        rho_nodes = 0.5 * (1.0 + np.tanh(arg))
        rho_weights = 0.25 * np.pi * h * ch_t / np.cosh(arg) ** 2

        mask = (rho_nodes > 0.0) & (rho_nodes < 1.0) & np.isfinite(rho_weights)
        rho_nodes = rho_nodes[mask]
        rho_weights = rho_weights[mask]

        for theta, wt in zip(th_nodes, th_weights):
            vals = np.array(
                [integrand(rho, theta) for rho in rho_nodes], dtype=np.float64
            )
            total += wt * np.dot(rho_weights, vals)
    elif radial_rule == "adaptive":
        for theta, wt in zip(th_nodes, th_weights):
            radial_val, _ = adaptive_quadrature(
                lambda rho: integrand(rho, theta),
                0,
                1,
                tol=1.0e-12,
                rtol=1.0e-12,
                maxiter=80,
                vec_func=False,
                miniter=3,
            )
            total += wt * float(radial_val)
    else:
        raise ValueError(f"unsupported radial_rule: {radial_rule}")

    return total, 0.0


def quadri_quad_duffy_radial(
    func,
    quadrilateral,
    singular_point,
    args=(),
    radial_rule="tanh-sinh",
    deg_theta=20,
    radial_quad_order=61,
    mp_dps=50,
):
    assert len(quadrilateral) == 4
    for p in quadrilateral:
        assert len(p) == 2

    trias = [
        (singular_point, quadrilateral[0], quadrilateral[1]),
        (singular_point, quadrilateral[1], quadrilateral[2]),
        (singular_point, quadrilateral[2], quadrilateral[3]),
        (singular_point, quadrilateral[3], quadrilateral[0]),
    ]

    val = np.zeros(4)
    err = np.zeros(4)
    for i in range(4):
        val[i], err[i] = tria_quad_duffy_radial(
            func,
            trias[i],
            args=args,
            radial_rule=radial_rule,
            deg_theta=deg_theta,
            radial_quad_order=radial_quad_order,
            mp_dps=mp_dps,
        )

    return np.sum(val), np.linalg.norm(err)


def box_quad_duffy_radial(
    func,
    a,
    b,
    c,
    d,
    singular_point,
    args=(),
    radial_rule="tanh-sinh",
    deg_theta=20,
    radial_quad_order=61,
    mp_dps=50,
):
    box = ((a, c), (b, c), (b, d), (a, d))

    if not isinstance(singular_point, tuple):
        singular_point = (singular_point[0], singular_point[1])

    singular_point = (max(singular_point[0], a), max(singular_point[1], c))
    singular_point = (min(singular_point[0], b), min(singular_point[1], d))

    return quadri_quad_duffy_radial(
        func,
        box,
        singular_point,
        args=args,
        radial_rule=radial_rule,
        deg_theta=deg_theta,
        radial_quad_order=radial_quad_order,
        mp_dps=mp_dps,
    )


def tria_quad_tanh_sinh_radial(func, tria, args=(), deg_theta=20, mp_dps=50):
    return tria_quad_duffy_radial(
        func,
        tria,
        args=args,
        radial_rule="tanh-sinh",
        deg_theta=deg_theta,
        radial_quad_order=61,
        mp_dps=mp_dps,
    )


def quadri_quad_tanh_sinh_radial(
    func, quadrilateral, singular_point, args=(), deg_theta=20, mp_dps=50
):
    return quadri_quad_duffy_radial(
        func,
        quadrilateral,
        singular_point,
        args=args,
        radial_rule="tanh-sinh",
        deg_theta=deg_theta,
        mp_dps=mp_dps,
    )


def box_quad_tanh_sinh_radial(
    func, a, b, c, d, singular_point, args=(), deg_theta=20, mp_dps=50
):
    return box_quad_duffy_radial(
        func,
        a,
        b,
        c,
        d,
        singular_point,
        args=args,
        radial_rule="tanh-sinh",
        deg_theta=deg_theta,
        mp_dps=mp_dps,
    )


def _duffy_regular_nodes_weights(dim_minus_one, deg_regular):
    if dim_minus_one == 0:
        return [np.array([], dtype=np.float64)], [1.0]

    nodes_1d, weights_1d = sp.special.p_roots(deg_regular)
    nodes_1d = 0.5 * (nodes_1d + 1.0)
    weights_1d = 0.5 * weights_1d

    grids = np.meshgrid(*([nodes_1d] * dim_minus_one), indexing="ij")
    wgrids = np.meshgrid(*([weights_1d] * dim_minus_one), indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], axis=-1)
    weights = np.prod(np.array(wgrids), axis=0).ravel()
    return nodes, weights


def _duffy_radial_nodes_weights(radial_rule, radial_quad_order, mp_dps):
    if radial_rule == "tanh-sinh-fast":
        n = max(3, int(radial_quad_order))
        h = 1.0 / np.sqrt(n)
        k = np.arange(-n, n + 1, dtype=np.float64)
        t = h * k
        sh_t = np.sinh(t)
        ch_t = np.cosh(t)
        arg = 0.5 * np.pi * sh_t
        rho_nodes = 0.5 * (1.0 + np.tanh(arg))
        rho_weights = 0.25 * np.pi * h * ch_t / np.cosh(arg) ** 2
        mask = (rho_nodes > 0.0) & (rho_nodes < 1.0) & np.isfinite(rho_weights)
        return rho_nodes[mask], rho_weights[mask]
    elif radial_rule == "tanh-sinh":
        old_dps = mpmath.mp.dps
        mpmath.mp.dps = mp_dps
        try:
            ts = mpmath.calculus.quadrature.TanhSinh(mpmath.mp)
            prec = int(np.log(10) / np.log(2) * mpmath.mp.dps)
            nodes = ts.calc_nodes(degree=max(3, int(radial_quad_order // 5)), prec=prec)
            rho_nodes = np.array([float(p[0]) for p in nodes], dtype=np.float64)
            rho_weights = np.array([float(p[1]) for p in nodes], dtype=np.float64)
        finally:
            mpmath.mp.dps = old_dps
        rho_nodes = 0.5 * (rho_nodes + 1.0)
        rho_weights = 0.5 * rho_weights
        return rho_nodes, rho_weights
    else:
        raise ValueError(f"unsupported radial_rule node set: {radial_rule}")


def box_quad_duffy_radial_nd(
    func,
    bounds,
    singular_point,
    args=(),
    radial_rule="tanh-sinh-fast",
    deg_regular=20,
    radial_quad_order=61,
    mp_dps=50,
):
    dim = len(bounds)
    assert len(singular_point) == dim
    singular_point = tuple(
        min(max(float(singular_point[i]), bounds[i][0]), bounds[i][1])
        for i in range(dim)
    )

    regular_nodes, regular_weights = _duffy_regular_nodes_weights(dim - 1, deg_regular)

    if radial_rule == "adaptive":
        radial_nodes = radial_weights = None
    else:
        radial_nodes, radial_weights = _duffy_radial_nodes_weights(
            radial_rule, radial_quad_order, mp_dps
        )

    total = 0.0
    from itertools import permutations, product

    for signs in product([-1.0, 1.0], repeat=dim):
        lengths = np.array(
            [
                singular_point[i] - bounds[i][0]
                if signs[i] < 0
                else bounds[i][1] - singular_point[i]
                for i in range(dim)
            ],
            dtype=np.float64,
        )
        if np.any(lengths <= 0):
            continue

        box_scale = float(np.prod(lengths))

        for perm in permutations(range(dim)):
            perm = list(perm)

            def eval_at_r(radial_r, tail_rs):
                rs = np.empty(dim, dtype=np.float64)
                rs[0] = radial_r
                if dim > 1:
                    rs[1:] = tail_rs

                u = np.empty(dim, dtype=np.float64)
                cumulative = 1.0
                for i, axis in enumerate(perm):
                    cumulative *= rs[i]
                    u[axis] = cumulative

                x = (
                    np.array(singular_point, dtype=np.float64)
                    + np.array(signs) * lengths * u
                )
                jac = box_scale * np.prod(
                    [rs[i] ** (dim - 1 - i) for i in range(dim - 1)]
                )
                val = np.asarray(func(*x, *args))
                val = float(val.reshape(-1)[0])
                prior = val * jac
                if not np.isfinite(prior):
                    if radial_r < 1.0e-14:
                        return 0.0
                    raise FloatingPointError((radial_r, tail_rs, prior, x))
                return prior

            if radial_rule == "adaptive":
                for tail_rs, w_tail in zip(regular_nodes, regular_weights):
                    radial_val, _ = adaptive_quadrature(
                        lambda rho: eval_at_r(rho, tail_rs),
                        0,
                        1,
                        tol=1.0e-12,
                        rtol=1.0e-12,
                        maxiter=80,
                        vec_func=False,
                        miniter=3,
                    )
                    total += w_tail * float(radial_val)
            else:
                for tail_rs, w_tail in zip(regular_nodes, regular_weights):
                    vals = np.array(
                        [eval_at_r(rho, tail_rs) for rho in radial_nodes],
                        dtype=np.float64,
                    )
                    total += w_tail * np.dot(radial_weights, vals)

    return total, 0.0


# }}}

# {{{ quadrature on a 2d box with a singular point inside


def box_quad(
    func,
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
    miniter=1,
):
    """Compute a singular 2D integral over an axis-aligned box.

    Integrate *func* on ``[a, b] x [c, d]`` using transformed Gaussian
    quadrature around *singular_point*.

    :arg func: callable integrand of two variables.
    :arg a: lower x bound.
    :arg b: upper x bound.
    :arg c: lower y bound.
    :arg d: upper y bound.
    :arg singular_point: singular point as ``(x, y)``.
    :arg args: extra positional arguments passed to *func*.
    :arg tol: absolute tolerance for adaptive quadrature.
    :arg rtol: relative tolerance for adaptive quadrature.
    :arg maxiter: maximum adaptive quadrature order.
    :arg vec_func: whether *func* accepts vectorized array inputs.
    :arg miniter: minimum adaptive quadrature order.
    :returns: ``(value, error_estimate)``.
    """
    box = ((a, c), (b, c), (b, d), (a, d))

    if not isinstance(singular_point, tuple):
        singular_point = (singular_point[0], singular_point[1])

    # When singular point is outside, project it onto the box bounday
    # This can import speed by not integrating around the actual singularity
    # when not necessary. (The splitting is still needed since it can be quite
    # close to singular).
    singular_point = (max(singular_point[0], a), max(singular_point[1], c))
    singular_point = (min(singular_point[0], b), min(singular_point[1], d))

    return quadri_quad(
        func, box, singular_point, args, tol, rtol, maxiter, vec_func, miniter
    )


# quadrilateral
def quadri_quad(
    func,
    quadrilateral,
    singular_point,
    args=(),
    tol=1.49e-08,
    rtol=1.49e-08,
    maxiter=50,
    vec_func=True,
    miniter=1,
):
    """Compute a singular 2D integral over a quadrilateral.

    :arg func: callable integrand of two variables.
    :arg quadrilateral: vertices ``((x1, y1), ..., (x4, y4))``.
    :arg singular_point: singular point as ``(x, y)``.
    :arg args: extra positional arguments passed to *func*.
    :arg tol: absolute tolerance for adaptive quadrature.
    :arg rtol: relative tolerance for adaptive quadrature.
    :arg maxiter: maximum adaptive quadrature order.
    :arg vec_func: whether *func* accepts vectorized array inputs.
    :arg miniter: minimum adaptive quadrature order.
    :returns: ``(value, error_estimate)``.
    """
    assert len(quadrilateral) == 4
    for p in quadrilateral:
        assert len(p) == 2

    # split the quadrilateral into four triangles
    trias = [
        (singular_point, quadrilateral[0], quadrilateral[1]),
        (singular_point, quadrilateral[1], quadrilateral[2]),
        (singular_point, quadrilateral[2], quadrilateral[3]),
        (singular_point, quadrilateral[3], quadrilateral[0]),
    ]

    for tria in trias:
        if not is_positive_triangle(tria):
            assert is_collinear(*tria)

    val = np.zeros(4)
    err = np.zeros(4)

    for i in range(4):
        val[i], err[i] = tria_quad(
            func, trias[i], args, tol, rtol, maxiter, vec_func, miniter
        )

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
