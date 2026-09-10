Installation and Setup
======================

Volumential uses ``pyproject.toml`` + ``uv`` as the primary dependency and
environment workflow.

Prerequisites
-------------

- Python ``3.12`` (the version CI tests; see the note below)
- OpenCL runtime (``pocl`` is the default tested backend)
- ``uv``

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
packages to their upstream Git sources for development.

Run Tests
---------

Run focused smoke tests:

.. code-block:: bash

   uv run pytest -q test/test_import.py
   uv run pytest -q test/test_duffy_tanh_sinh.py

Run the full suite:

.. code-block:: bash

   uv run pytest

Include long-running checks:

.. code-block:: bash

   uv run pytest --longrun

Remote Environment
------------------

Use the same setup steps on remote machines used for heavier experiments
(for example ``ipa``) so local and remote behavior stays aligned:

.. code-block:: bash

   ssh ipa
   git clone <repo-url>
   cd volumential
   micromamba create -n volumential-dev -c conda-forge -c nodefaults \
     python=3.12 pyopencl pocl scipy numpy
   micromamba activate volumential-dev
   uv sync --active --extra test --extra doc

For an expanded local/remote workflow reference, see ``DEVELOPMENT.md``.
