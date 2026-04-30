Periodic Tail Hybrid Experiment (2D)
====================================

This page documents the periodic Laplace hybrid workflow implemented in:

- ``examples/periodic_tail_hybrid_experiment.py``
- ``examples/periodic_tail_hybrid_sweep.py``

Problem Setup
-------------

Let the periodic cell be :math:`\Omega=[0,L)^2` with source samples
:math:`\{(y_j,q_j)\}_{j=1}^N` (here :math:`q_j` already includes quadrature
weights). The 2D Laplace kernel used in code is

.. math::

   G(r) = -\frac{1}{4\pi}\log(r\cdot r)
        = -\frac{1}{2\pi}\log|r|.

The periodic potential is formally

.. math::

   u_{\mathrm{per}}(x)
   = \sum_{j=1}^N q_j \sum_{n\in\mathbb{Z}^2} G\bigl(x-y_j-Ln\bigr).

For Poisson periodicity we enforce neutrality:

.. math::

   \sum_{j=1}^N q_j = 0.

Hybrid Decomposition
--------------------

With near-image radius :math:`R_{\mathrm{near}}` (in :math:`\ell_\infty`
lattice shells), write

.. math::

   u_{\mathrm{per}} = u_0 + u_{\mathrm{near}} + u_{\mathrm{tail}},

where

.. math::

   u_0(x) = \sum_j q_j G(x-y_j),

.. math::

   u_{\mathrm{near}}(x)
   = \sum_j q_j \sum_{1\le \|n\|_\infty\le R_{\mathrm{near}}}
      G(x-y_j-Ln),

.. math::

   u_{\mathrm{tail}}(x)
   = \sum_j q_j \sum_{\|n\|_\infty>R_{\mathrm{near}}}
      G(x-y_j-Ln).

Implementation mapping:

- :math:`u_0` from central-cell volumential FMM,
- :math:`u_{\mathrm{near}}` from direct shell sums,
- :math:`u_{\mathrm{tail}}` from polynomial tail coefficients.

The shifted-image P2P path is intentionally disabled in this workflow.

Tail-Coefficient Derivation
---------------------------

Choose expansion center :math:`c=(L/2,L/2)`, define
:math:`t=x-c` and :math:`s_j=y_j-c`. For each far image :math:`n`, expand

.. math::

   G(t-s_j-Ln)
   = \sum_{\nu\in\mathbb{N}_0^2}
     \frac{1}{\nu!} D^\nu G(-Ln) (t-s_j)^\nu.

Using multi-index splitting :math:`\nu=\alpha+\beta` and
:math:`(t-s_j)^\nu = \sum_{\alpha+\beta=\nu}
\frac{\nu!}{\alpha!\beta!} t^\beta (-s_j)^\alpha`,

.. math::

   G(t-s_j-Ln)
   = \sum_{\alpha,\beta\in\mathbb{N}_0^2}
     \frac{(-1)^{|\alpha|}}{\alpha!\,\beta!}
     D^{\alpha+\beta}G(-Ln)\, t^\beta s_j^\alpha.

Define lattice derivative sums and source moments:

.. math::

   S_\nu
   := \sum_{\|n\|_\infty>R_{\mathrm{near}}} D^\nu G(-Ln),
   \qquad
   M_\alpha := \sum_{j=1}^N q_j s_j^\alpha.

Then

.. math::

   u_{\mathrm{tail}}(t)
   = \sum_{\beta} t^\beta \sum_{\alpha} T_{\beta,\alpha} M_\alpha,

with

.. math::

   T_{\beta,\alpha}
   = \frac{(-1)^{|\alpha|}}{\alpha!\,\beta!} S_{\alpha+\beta}.

This is exactly the coefficient structure used in the experiment script.

How :math:`S_\nu` Is Computed
-----------------------------

1. **Parity cancellation**

   By lattice symmetry :math:`n\leftrightarrow -n`, odd total derivative order
   vanishes:

   .. math::

      |\nu| \text{ odd} \implies S_\nu=0.

2. **Low-order sums (:math:`|\nu|\le 2`)**

   Conditional convergence is handled with Gaussian-window regularization:

   .. math::

      S_\nu^{(\eta)}
      := \sum_{\|n\|_\infty>R_{\mathrm{near}}}
         D^\nu G(-Ln)\,e^{-\eta|Ln|^2}.

   Evaluate multiple small :math:`\eta` values and fit/extrapolate
   :math:`\eta\to 0^+` to recover :math:`S_\nu`.

3. **Higher-order sums (:math:`|\nu|>2`)**

   Absolute convergence allows either:

   - hard-cutoff shell sums with Richardson/inverse-even-power extrapolation,
   - high-precision Eisenstein-based evaluation (default path), with explicit
     subtraction of the near-prefix shells.

For hard cutoffs :math:`R`:

.. math::

   S_\nu(R)
   = \sum_{R_{\mathrm{near}}<\|n\|_\infty\le R} D^\nu G(-Ln)
   = S_\nu + a_1 R^{-2} + a_2 R^{-4} + \cdots,

which motivates the inverse-even-power fit used by the script.

Gauge Derivation (Why Dipole Correction Works)
----------------------------------------------

The square-sum lattice convention and the Fourier periodic Green convention
differ by an affine gauge:

.. math::

   u_{\mathrm{lattice}}(x)-u_{\mathrm{spec}}(x)
   = c_0 + a\cdot(x-c).

For neutral sources, slope is set by dipole moment
:math:`m=\sum_j q_j (y_j-c)` and area :math:`A=L^2`:

.. math::

   a = -\frac{m}{2A}.

So the physics-based correction is

.. math::

   u_{\mathrm{dip}}(x)
   = u_{\mathrm{hybrid}}(x) - \frac{1}{2A}m\cdot(x-c),

plus an arbitrary constant (the script optionally removes mean value on the
probe grid). This is the reported ``dipole-corrected`` metric.

The ``affine-corrected`` metric additionally solves

.. math::

   \min_{c_0,a_x,a_y}
   \|u_{\mathrm{hybrid}} + c_0 + a_x(x-c_x)+a_y(y-c_y)-u_{\mathrm{ref}}\|_2,

which is a looser gauge alignment than dipole-only correction.

Diagnostics
-----------

- **Reference error**

  .. math::

     \mathrm{rel\_L2}(v,v_\star)=\frac{\|v-v_\star\|_2}{\|v_\star\|_2}.

- **PDE residual (interior patch)**

  .. math::

     r = -\Delta u_{\mathrm{hybrid}} - \rho.

- **Periodicity jumps (translated-pair check)**

  .. math::

     J_u^{(e_i)}(x)=u(x+Le_i)-u(x),
     \qquad
     J_{\nabla u}^{(e_i)}(x)=\nabla u(x+Le_i)-\nabla u(x).

By default the script evaluates these on translated opposite faces, which is the
correct periodicity test.

Precision and Stability Knobs
-----------------------------

Optional compensated/block accumulation controls:

- direct image sums:
  ``--direct-sum-source-block-size``,
  ``--direct-sum-extended-precision``
- spectral reference:
  ``--spectral-compensated``,
  ``--spectral-source-block-size``,
  ``--spectral-accum-extended-precision``

These are for sensitivity studies and can increase runtime substantially.

Current Practical Accuracy Floor
--------------------------------

In current float64 runs, increasing ``nlevels`` improves gauge-aligned errors
strongly at first, then levels off near a few :math:`10^{-12}` for the tested
setup. This is acceptable for current validation goals and serves as a
reasonable stopping point before broader cleanup and 3D extension work.

Single-Run Experiment
---------------------

Example:

.. code-block:: bash

   uv run python examples/periodic_tail_hybrid_experiment.py \
     --seed 17 --q-order 5 --nlevels 6 --fmm-order 20 \
     --near-radius 2 --max-order 20 --kmax 64

Useful diagnostics:

- spectral-reference error (raw / dipole-corrected / affine-corrected),
- PDE residual check on a calculus patch,
- periodicity jumps on translated face pairs.

Sweep Script
------------

For parameter/seed sweeps:

.. code-block:: bash

   uv run python examples/periodic_tail_hybrid_sweep.py \
     --q-order 5 --nlevels 6 --fmm-order 20 --kmax 64 \
     --seeds 3,7,11,17,23 --near-radii 1,2 --max-orders 12,16,20

The sweep reports mean/std/max statistics and tracks best settings by
dipole-corrected relative L2 error.
