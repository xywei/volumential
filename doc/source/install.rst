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
   conda install matplotlib mayavi pyvtk pyvisfile

.. note::

   Deal.II_ is needed for adaptive mesh refinement. For now, Deal.II has to be manually compiled. It is a work in progress to enable installing Deal.II via conda at https://github.com/conda-forge/staged-recipes/pull/8162.

   .. _Deal.II: https://www.dealii.org/

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

Optionally, if using ``meshgen`` via Deal.II, compile the ``meshgen`` module under ``contrib/meshgen11_dealii``. 

.. code-block:: bash

   cd contrib/meshgen11_dealii
   DEAL_II_DIR=/path/to/deaii/installation make

After installation, checkout ``examples/`` for example usage.


