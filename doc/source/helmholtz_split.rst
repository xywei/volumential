Helmholtz Split Formulation
===========================

This page documents the Helmholtz near-field split used in
:mod:`volumential.expansion_wrangler_fpnd`.

The split applies to list1 (neighbor-box) interactions only.
Far-field FMM interactions continue to use the Helmholtz kernel directly.

Decomposition
-------------

For neighbor interactions, we write

.. math::

    G_k(r) = G_0(r) + S_p(r) + R_p(r),

where:

- :math:`G_0` is the Laplace kernel term, handled by standard Laplace
  near-field tables.
- :math:`S_p` is a finite sum of *non-smooth* split terms, handled by
  additional prebuilt near-field term tables.
- :math:`R_p` is the analytic remainder, evaluated online with P2P.

Hence the neighbor potential contribution is

.. math::

    \mathcal{N}_k[\rho]
    = \mathcal{N}_{0,\mathrm{table}}[\rho]
    + \mathcal{N}_{S_p,\mathrm{table}}[\rho]
    + \mathcal{N}_{R_p,\mathrm{p2p}}[\rho].

Reusable Split-Term Notation
----------------------------

To avoid duplicating implementation logic, both 2D and 3D paths are written in
the same abstract form:

.. math::

    G_k(r) - G_0(r)
    = C_k + \sum_{n\ge 1} \beta_n(k)\,\phi_n(r) + \Psi_k(r),

where:

- :math:`\phi_n` are the non-smooth radial basis terms that get table support,
- :math:`\beta_n(k)` are runtime coefficients,
- :math:`\Psi_k` is the remaining smooth/higher-order part evaluated online.

For split order :math:`p`, Volumential tables
:math:`\{\phi_1,\ldots,\phi_{p-1}\}` and evaluates

.. math::

    R_p(r) = C_k + \sum_{n\ge p} \beta_n(k)\,\phi_n(r) + \Psi_k(r)

in the online correction path.

This is exactly the same cache/reuse pattern used for Helmholtz and Yukawa;
Yukawa uses the mapping :math:`k=i\lambda` in the coefficient formulas.

2D Expansion
------------

With

.. math::

    G_k(r) = \frac{i}{4} H_0^{(1)}(k r),
    \qquad
    G_0(r) = -\frac{1}{2\pi}\log r,

the difference has the local expansion

.. math::

    G_k(r) - G_0(r)
    = c_0(k)
    + \sum_{n=1}^{\infty}\left[a_n(k) r^{2n}\log r + b_n(k) r^{2n}\right],

with

.. math::

    c_0(k) = \frac{i}{4} - \frac{\log(k/2)+\gamma}{2\pi},

.. math::

    a_n(k)
    = -\frac{1}{2\pi}
      \frac{(-1)^n}{(n!)^2}\left(\frac{k^2}{4}\right)^n,

.. math::

    b_n(k)
    = \frac{(-1)^n}{(n!)^2}\left(\frac{k^2}{4}\right)^n
      \left[
        \frac{H_n - (\log(k/2)+\gamma)}{2\pi} + \frac{i}{4}
      \right].

These coefficients come directly from the classical small-argument expansions:

.. math::

    J_0(z) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(n!)^2}\left(\frac{z^2}{4}\right)^n,

.. math::

    Y_0(z) = \frac{2}{\pi}\left(\log(z/2)+\gamma\right)J_0(z)
    - \frac{2}{\pi}\sum_{n=1}^{\infty} H_n
      \frac{(-1)^n}{(n!)^2}\left(\frac{z^2}{4}\right)^n,

with :math:`H_0^{(1)}(z)=J_0(z)+iY_0(z)` and :math:`z=kr`.

Using the reusable notation above:

- :math:`\phi_n(r)=r^{2n}\log r`,
- :math:`\beta_n(k)=a_n(k)`,
- :math:`\Psi_k(r)=\sum_{n\ge 1} b_n(k)r^{2n}`,
- :math:`C_k=c_0(k)`.

Split order :math:`p` means:

- pretabulate non-smooth terms :math:`a_n r^{2n}\log r`,
  :math:`n=1,\dots,p-1`;
- keep all smooth polynomial terms :math:`c_0 + \sum b_n r^{2n}` in the online
  remainder;
- keep higher non-smooth terms :math:`a_n r^{2n}\log r` for :math:`n\ge p`
  in the online remainder.

So for 2D:

- ``split_order=1`` extracts no non-smooth extra terms;
- ``split_order=2`` extracts :math:`r^2\log r`;
- ``split_order=3`` extracts :math:`r^2\log r` and :math:`r^4\log r`, etc.

2D Empirical Notes
------------------

Manufactured-solution sweeps (Gaussian exact solution) in 2D show:

- ``p=2`` usually gives the dominant gain over ``p=1``;
- ``p=3`` and ``p=4`` can provide additional reduction, but the benefit can be
  small once other error sources dominate;
- for tuned cases, ``m=q`` works with higher split orders and reaches high
  accuracy without requiring ``m=q+1``.

Representative tuned run (PoCL GPU, ``q=7``, ``nlevels=4``, ``fmm=20``,
``k=20``, ``alpha=120``, ``m=q=7``):

=================  ========================
split order ``p``  relative L2 error
=================  ========================
1                  ``9.725554e-05``
2                  ``1.093584e-06``
3                  ``7.819492e-07``
4                  ``7.784375e-07``
5                  ``7.786645e-07``
6                  ``7.783663e-07``
=================  ========================

High-accuracy run (PoCL CPU, ``q=15``, ``nlevels=6``, ``fmm=21``, ``k=8``,
``alpha=120``, ``m=q=15``):

=================  ========================
split order ``p``  relative L2 error
=================  ========================
1                  ``3.235351e-09``
2                  ``1.135107e-11``
3                  ``1.135099e-11``
4                  ``1.135099e-11``
5                  ``1.135099e-11``
=================  ========================

In this regime, ``p=2`` captures most of the split truncation benefit and
``p>=3`` is near saturation for total error.

3D Expansion
------------

With

.. math::

    G_k(r) = \frac{e^{ikr}}{4\pi r},
    \qquad
    G_0(r) = \frac{1}{4\pi r},

we have

.. math::

    G_k(r)-G_0(r)
    = \sum_{n=1}^{\infty} c_n(k) r^{n-1},
    \qquad
    c_n(k)=\frac{(ik)^n}{4\pi\,n!}.

Equivalently, from :math:`e^{ikr}=\sum_{n\ge 0}(ikr)^n/n!`:

.. math::

    \frac{e^{ikr}-1}{4\pi r}
    = \sum_{n=1}^{\infty}\frac{(ik)^n}{4\pi\,n!}r^{n-1}.

Terms :math:`r^{2j-1}` are the non-smooth radial powers at the origin.
Split order :math:`p` extracts

.. math::

    \sum_{j=1}^{p-1} \frac{(ik)^{2j}}{4\pi\,(2j)!} r^{2j-1}

into prebuilt split tables, while the online remainder keeps the constant,
even powers, and higher odd powers.

Using the reusable notation above:

- :math:`\phi_j(r)=r^{2j-1}` (odd-power branch),
- :math:`\beta_j(k)=(ik)^{2j}/(4\pi(2j)!)`,
- :math:`\Psi_k` collects constant/even-power terms and odd powers
  :math:`j\ge p`.

High-Reference Wideband Convergence Snapshot
--------------------------------------------

Copy of the IPA table used in PR #79 for split-auto defaults
(``helmholtz_split_order="auto"`` with auto smooth quadrature), reporting
``rel_vs_direct_high``.

.. list-table::
   :header-rows: 1
   :widths: 14 5 6 6 8 8 6 6 14

   * - Kernel
     - Dim
     - Re(k)
     - Im(k)
     - rho_real
     - rho_imag
     - auto p
     - auto m
     - rel_vs_direct_high
   * - Helmholtz
     - 2
     - 2.0
     - 0
     - 0.5
     - 0
     - 3
     - 7
     - 2.3436e-08
   * - Helmholtz
     - 2
     - 8.0
     - 0
     - 2.0
     - 0
     - 5
     - 9
     - 7.9173e-08
   * - Helmholtz
     - 2
     - 16.0
     - 0
     - 4.0
     - 0
     - 6
     - 11
     - 3.3398e-07
   * - Helmholtz
     - 2
     - 32.0
     - 0
     - 8.0
     - 0
     - 7
     - 14
     - 2.6230e-06
   * - Helmholtz
     - 3
     - 2.0
     - 0
     - 0.5
     - 0
     - 3
     - 5
     - 2.2215e-12
   * - Helmholtz
     - 3
     - 4.0
     - 0
     - 1.0
     - 0
     - 4
     - 6
     - 1.3885e-13
   * - Helmholtz
     - 3
     - 6.0
     - 0
     - 1.5
     - 0
     - 5
     - 7
     - 5.5698e-13
   * - Helmholtz
     - 3
     - 8.0
     - 0
     - 2.0
     - 0
     - 5
     - 7
     - 3.4457e-12
   * - Helmholtz
     - 3
     - 10.0
     - 0
     - 2.5
     - 0
     - 6
     - 8
     - 1.9042e-11
   * - Helmholtz
     - 3
     - 12.0
     - 0
     - 3.0
     - 0
     - 6
     - 8
     - 1.9167e-10
   * - Helmholtz
     - 3
     - 16.0
     - 0
     - 4.0
     - 0
     - 6
     - 9
     - 1.1670e-08
   * - Yukawa
     - 2
     - 0
     - 8
     - 0
     - 2.0
     - 4
     - 8
     - 1.8611e-09
   * - Yukawa
     - 2
     - 0
     - 16
     - 0
     - 4.0
     - 5
     - 9
     - 6.1665e-08
   * - Yukawa
     - 2
     - 0
     - 24
     - 0
     - 6.0
     - 6
     - 12
     - 1.2716e-07
   * - Yukawa
     - 2
     - 0
     - 32
     - 0
     - 8.0
     - 6
     - 14
     - 6.6297e-08
   * - Yukawa
     - 3
     - 0
     - 2
     - 0
     - 0.5
     - 2
     - 4
     - 4.8224e-07
   * - Yukawa
     - 3
     - 0
     - 4
     - 0
     - 1.0
     - 3
     - 5
     - 8.7211e-07
   * - Yukawa
     - 3
     - 0
     - 8
     - 0
     - 2.0
     - 4
     - 6
     - 2.0533e-06
   * - Yukawa
     - 3
     - 0
     - 12
     - 0
     - 3.0
     - 5
     - 7
     - 3.8141e-06
   * - Yukawa
     - 3
     - 0
     - 16
     - 0
     - 4.0
     - 5
     - 7
     - 6.1529e-06

Design Principles (Why This Layer Exists)
-----------------------------------------

The split implementation is organized around a **kernel-agnostic basis cache**
plus **runtime kernel coefficients**.

- Split basis tables (Laplace + split terms such as ``power``/``power_log``)
  depend on geometry/discretization, not on runtime wave number.
- Helmholtz/Yukawa parameter values are applied online in split coefficients and
  smooth remainders.
- This enables broad reuse across wave numbers while preserving correctness.

In contrast, direct (non-split) parameterized kernel tables remain
parameter-specific and are validated against requested kernel parameters when
loaded from cache.

Automatic Regime Planner
------------------------

For wide ranges of :math:`k` and mesh sizes, one fixed split order is often
suboptimal. Volumential now provides a lightweight automatic planner that
chooses split order from the dimensionless local scale

.. math::

    \rho_{\max} = \max_{\ell\in\text{active levels}} |k| h_\ell,

where :math:`h_\ell` is source-box extent at level :math:`\ell`.

For Yukawa split mode, the planner uses the internal mapping
:math:`k = i\,\lambda`, so :math:`|k|=|\lambda|` in the same formula.

Default planner policy:

- ``order_min=2``
- ``order_max=12``
- geometric rho ladders with ``rho_base_real=0.25`` and ``rho_base_imag=0.5``
- hard-real smooth trigger ``smooth_quad_order_hard_rho_real=3.0``

This yields thresholds like ``(0.25, 0.5, 1, 2, 4, 8, 16)`` for
:math:`|\Re(k)|h` and ``(0.5, 1, 2, 4, 8, 16, 32)`` for
:math:`|\Im(k)|h`, with candidate orders ``2..12``. This default is tuned to
be more accuracy-forward in easy regimes while keeping split-order selection
purely ladder-based.

Runtime Configuration Knobs
---------------------------

You may enable auto selection by either:

- passing ``helmholtz_split_order="auto"``, or
- passing ``helmholtz_split_auto_config={"enabled": True, ...}``.

Supported planner config keys:

- ``rho_thresholds``: sequence of increasing boundaries
- ``rho_thresholds_real`` / ``rho_thresholds_imag``: optional per-component
  threshold ladders (override defaults)
- ``orders``: sequence of selected orders (length = ``len(rho_thresholds)+1``)
- ``order_min`` / ``order_max``: optional clamps
- ``rho_base_real`` / ``rho_base_imag``: base values for default geometric
  threshold ladders for :math:`|\Re(k)|h` and :math:`|\Im(k)|h`
- ``rho_imag_split_max``: maximum :math:`|\Im(k)|h` for default split mode
- ``disable_split_if_outside_coverage``: when :math:`|\Im(k)|h` exceeds
  ``rho_imag_split_max``, Volumential currently emits warnings and keeps split
  evaluation enabled (direct fallback is not performed because it would require
  matching direct near-field tables)

  By default, ``disable_split_if_outside_coverage`` is ``False``. Auto
  selection keeps split enabled, clamps to the highest configured split order,
  and emits warnings when :math:`|\Im(k)|h` exceeds configured coverage.

- ``smooth_quad_order_min``: floor for smooth quadrature order
- ``smooth_quad_order_per_order``: increment per additional split order above 1
  for easy/moderate attenuation regimes (default ``1``)
- ``smooth_quad_order_per_order_hard``: increment per additional split order above
  1 for hard attenuation regimes (default ``1``)
- ``smooth_quad_order_hard_rho_imag``: hard-regime trigger on
  :math:`|\Im(k)|h` for switching from
  ``smooth_quad_order_per_order`` to
  ``smooth_quad_order_per_order_hard`` (default ``4.0``)
- ``smooth_quad_order_hard_rho_real``: hard-regime trigger on
  :math:`|\Re(k)|h` for switching from
  ``smooth_quad_order_per_order`` to
  ``smooth_quad_order_per_order_hard`` (default ``3.0``)
- ``smooth_quad_order_rho_boost_start``: start of direct
  :math:`|\Im(k)|h`-based smooth-order boost (default same as
  ``smooth_quad_order_hard_rho_imag``)
- ``smooth_quad_order_rho_boost_scale``: boost slope in
  :math:`\Delta m \approx \lceil \mathrm{scale}\cdot(
  \rho_{\mathrm{imag}}-\rho_0)\rceil`
  (default ``1.0``)
- ``smooth_quad_order_rho_boost_cap``: optional cap on direct rho-based boost
- ``smooth_quad_order_real_boost_start``: start of direct
  :math:`|\Re(k)|h`-based smooth-order boost (default same as
  ``smooth_quad_order_hard_rho_real``)
- ``smooth_quad_order_real_boost_scale``: boost slope in
  :math:`\Delta m \approx \lceil \mathrm{scale}\cdot(
  \rho_{\mathrm{real}}-\rho_0)\rceil`
  (default ``0.5``)
- ``smooth_quad_order_real_boost_cap``: optional cap on direct
  :math:`|\Re(k)|h`-based smooth-order boost
- ``smooth_quad_order_max``: optional cap on final auto-selected smooth order
- ``power_log_single_table_beta_mode``: backend for the
  :math:`(\log \alpha_\ell)r^{2n}` correction when 2D ``power_log`` terms use
  one reference table. ``"p2p"`` (default) evaluates this correction online
  with the active smooth-correction source layout (including oversampled
  ``m>q`` sources). ``"table"`` evaluates it via split-power tables.

When auto mode is active and ``helmholtz_split_smooth_quad_order`` is not set,
the smooth quadrature order is chosen from these knobs instead of forcing
``m=q``. The default policy is accuracy-forward in easy regimes
(``smooth_quad_order_per_order=1``) while retaining stronger smooth-quadrature
growth in hard attenuation regimes via
``smooth_quad_order_per_order_hard=1`` and an additional direct
:math:`|\Im(k)|h`-based boost, while also allowing extra smooth-order growth
for high real-frequency regimes via :math:`|\Re(k)|h`-based triggers/boosts.

Yukawa and Mixed-Complex Parameters
-----------------------------------

- Yukawa split reuse follows the same basis-table mechanism as Helmholtz.
- Current Yukawa split path requires real ``lam``.
- For mixed complex wave numbers (nonzero real+imag parts), use Helmholtz
  kernels directly.

Implementation Notes
--------------------

- ``helmholtz_split_smooth_quad_order`` defaults to ``m=q`` when auto mode is
  inactive. In auto mode, the default smooth order follows the planner policy
  described above.
- split-order-1 now defaults to the analytic series-remainder kernel path,
  avoiding runtime Helmholtz-minus-Laplace singular subtraction.
- the historical split-order-1 subtraction path is still available for
  internal checks via
  ``helmholtz_split_order1_legacy_subtraction=True``.
- smooth-node overlap checks (``m>q`` with shared Gauss nodes) are enforced for
  the legacy subtraction path. The default series-remainder path allows overlap.
- in 2D, removable :math:`r^{2n}\log r` singularities in the remainder are
  guarded in-kernel at very small :math:`r`.
- split-term tables are managed under the same
  :class:`volumential.table_manager.NearFieldInteractionTableManager` umbrella
  cache as the Laplace near-field tables.
- for 2D :math:`r^{2n}\log r` split terms, a single reference table is
  scaled across levels. The non-homogeneous scaling remainder

  .. math::

      (\log \alpha_\ell) r^{2n}

  is added online by folding :math:`\log \alpha_\ell` into source strengths
  and evaluating the matching :math:`r^{2n}` correction kernel. By default this
  uses an online P2P correction with the same source layout as the active
  smooth-correction path (including oversampled ``m>q`` sources), which avoids
  prebuilding extra split-power tables and typically improves list1 throughput
  on GPU. The alternative table-based correction path can be selected with
  ``helmholtz_split_auto_config={"power_log_single_table_beta_mode": "table"}``.
  Here :math:`\alpha_\ell = h_\ell / h_{\mathrm{ref}}` and :math:`h_\ell` is the
  source-box extent.
- when self interactions are excluded in correction P2P, the finite
  :math:`r\to0` diagonal limit of :math:`G_k-G_0` is added back analytically.

3D Empirical Notes
------------------

Manufactured-solution sweeps (Gaussian exact solution) in 3D show a consistent
pattern:

- increasing split order from ``p=1`` to ``p=2`` yields a strong reduction in
  total error;
- improvements from ``p>=3`` are usually much smaller and can be masked by
  non-split errors (FMM/truncation/interpolation/table-accuracy floors);
- with tighter table build settings, small ``p=2 -> p=3 -> p=4`` improvements
  become visible in total error.

Representative run (PoCL CPU, ``q=7``, ``nlevels=4``, ``fmm=20``,
``k=14``, ``alpha=80``, ``smooth_q=11``):

=================  ========================
split order ``p``  relative L2 error
=================  ========================
1                  ``5.545779e-06``
2                  ``2.424334e-06``
3                  ``2.423980e-06``
4                  ``2.423978e-06``
=================  ========================

To reproduce and plot 3D split-order convergence, use
``examples/helmholtz3d_split_p_convergence.py``.
