Installation and Setup
======================

This section introduces how to setup a ``conda`` environment with everything needed to perform volume potential evaluations.

Conda Setup
-----------

Download and install ``miniconda`` or ``anaconda``. For ``miniconda``, follow the instructions at https://docs.conda.io/en/latest/miniconda.html. For the impatient, the following script installs and creates a ``conda`` environment for later use.

.. code-block:: bash

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   sh Miniconda3-latest-Linux-x86_64.sh
   . ~/miniconda3/etc/profile.d/conda.sh
   conda create -n scicomp
   conda activate scicomp

Then, make the ``conda-forge`` channel default.

.. code-block:: bash

   conda config --add channels conda-forge 

Install Dependencies
--------------------

Install dependencies using ``conda`` (after activating the newly created environment).

.. code-block:: bash

   conda install python=3
   conda install pyopencl pocl sympy scipy 
   conda install matplotlib mayavi pyvtk

Install ``dealii`` as well for additional features.

.. note::

   Deal.II is used only for adaptive mesh refinement (AMR).

.. code-block:: bash

   conda install tbb-devel dealii

And here are some optional dependencies. Install these if you intend to develop the library or build this documentation.

.. code-block:: bash

   conda install flake8 pylint mypy pylama
   conda install pytest sphinx sphinx_rtd_theme

Note that if you are using ``jupyterlab`` or ``jupyter`` notebook, you need to also install them inside the ``conda`` environment. For example,

.. code-block:: bash

   # nodejs is needed for installing extensions
   conda install jupyter jupyterlab nodejs

   jupyter-labextension install jupyterlab-jupytext
   jupyter-labextension install jupyterlab_vim

Install Volumential
-------------------

We recommend downloading :py:mod:`volumential` using ``git``.

.. code-block:: bash

   # clone via SSH
   git clone git@gitlab.tiker.net:xywei/volumential.git

   # clone via HTTPS
   git clone https://gitlab.tiker.net/xywei/volumential.git

Then finish installation by running

.. code-block:: bash

   cd volumential

   # installs volumential along with some additional
   # dependencies not present in conda-forge
   pip install -r requirements.txt

Compile the AMR Module
----------------------

The AMR module is implemented based on ``dealii``.
To use it, compile the ``meshgen`` module under ``contrib/meshgen11_dealii``. 

.. code-block:: bash

   cd contrib/meshgen11_dealii
   git submodule update --init --recursive
   make

If the build process fails with the error message
``The keyword signature for target_link_libraries has already been used with
the target ...``, try editing ``CMakeLists.txt`` and change line 52
from ``if(TRUE)`` to ``if(FALSE)``. Then remove the ``build`` directory
and re-run ``make``.

If you are using manually compiled ``dealii`` instead of the one from
``conda-forge``, set the ``DEAL_II_DIR`` environment variable to
its installation path before calling ``make``.

Alternatively, there is another implementation of the AMR module
using ``boost::python`` under ``contrib/meshgen_dealii``. The compilation
process is similar. It is in deprecated status. But if you have troubles
compiling the ``pybind11`` one, it may worth a try.

After installation, checkout ``examples/`` for example usage.

Update the Installation
-----------------------

The update process is simple using the ``requirements.txt``.

.. code-block:: bash

   git pull
   git submodule update --init --recursive
   pip install -U -r requirements.txt
