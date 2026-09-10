Installation and Setup
======================

Volumential uses ``pyproject.toml`` + ``uv`` as the primary dependency and
environment workflow.

Prerequisites
-------------

- Python ``3.12`` (the version CI tests; see the note below)
- OpenCL runtime (``pocl`` is the default tested backend)
- ``uv``
- ``gfortran`` and ``ninja``, only for the optional ``fmmlib`` extra

.. note::

   ``requires-python`` in ``pyproject.toml`` is still ``>=3.11``, and nothing
   in Volumential itself needs 3.12. CI pins 3.12 only because ``loopy``
   currently imports ``override`` from the standard-library ``typing``
   module, which gained it in 3.12; under 3.11 ``import loopy`` fails
   outright. A 3.11 environment is therefore unsupported in practice until
   that upstream import moves to ``typing_extensions``.

Install ``uv``
--------------

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Create a Base OpenCL Environment
--------------------------------

Use micromamba/conda to provide OpenCL runtime dependencies:

.. code-block:: bash

   micromamba create -n volumential-dev -c conda-forge -c nodefaults \
     python=3.12 pyopencl pocl scipy numpy
   micromamba activate volumential-dev

Install Volumential with uv
---------------------------

Clone and sync dependencies from ``pyproject.toml``:

.. code-block:: bash

   git clone https://github.com/xywei/volumential.git
   cd volumential
   uv sync --active --extra test --extra doc

The ``tool.uv.sources`` table in ``pyproject.toml`` points the inducer-stack
packages at their upstream Git sources, and ``uv.lock`` records the resolved
commits. Released wheels of those packages are not suitable for adaptive-tree
experiments; see ``DEVELOPMENT.md`` for the reasoning and for the traversal
sanity check that a freshly provisioned environment must pass.

Add the FMMLib backend with

.. code-block:: bash

   uv pip install --active \
       "pyfmmlib @ git+https://github.com/inducer/pyfmmlib.git"

which builds ``pyfmmlib`` from upstream ``main``, where OpenMP and the batched
``formmp`` wrappers live. Both need a host with ``gfortran`` and ``ninja``.

.. note::

   Not ``uv sync --extra fmmlib``. Unlike the inducer-stack packages above,
   ``pyfmmlib`` has no ``tool.uv.sources`` entry, so the ``fmmlib`` extra
   resolves its declared ``pyfmmlib>=2024.1`` to the PyPI ``2024.1.1``
   release -- which has neither feature, and
   ``FPNDFMMLibExpansionWrangler`` then falls back to its serial per-box
   path without complaining, making any FMMLib timing misleading.
   Pull request 135 adds the Git source on ``main``; once that has landed
   here, ``uv sync --active --extra fmmlib`` becomes the recipe and this
   note can go. ``DEVELOPMENT.md`` carries the verification commands that
   confirm which build you got.

Select an OpenCL Platform
-------------------------

.. code-block:: bash

   export PYOPENCL_CTX=portable:0
   export PYOPENCL_TEST=portable:0

On NixOS, also point ICD discovery at a single vendor directory, otherwise
``pyopencl`` fails with ``PLATFORM_NOT_FOUND_KHR``:

.. code-block:: bash

   export OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors
   export OPENCL_VENDOR_PATH=/run/opengl-driver/etc/OpenCL/vendors

Run Tests
---------

Run focused smoke tests:

.. code-block:: bash

   uv run pytest -q test/test_import.py
   uv run pytest -q test/test_public_surface.py
   uv run pytest -q test/test_duffy_tanh_sinh.py

Run the full suite:

.. code-block:: bash

   uv run pytest

Include long-running checks:

.. code-block:: bash

   uv run pytest --longrun

Remote Environment
------------------

Heavier experiments run on a suitable, currently idle remote machine, using the
same recipe as above so that local and remote behavior stay aligned. Keep host
names and paths out of committed material.

For an expanded local/remote workflow reference, see ``DEVELOPMENT.md``.
