# Mesh generation via Deal.II

**This module is deprecated and no longer maintained. Use `meshgen11\_dealii` instead**

`meshgen` generates a set of quadrature points and weights using `Deal.II`

## Build

To build `meshgen`, you need:

- Python 3.6 or later
- Boost 1.65.0 or later, with python & numpy extensions
- Deal.II 8.5.0 or later

(To avoid possible incompatibilities, make sure to build Deal.II using Boost
1.65.0 and Python 3.6)

To build, first edit `CMakeLists.txt` to replace the include paths and library
paths, (unfortunately cmake support for miniconda packages is not so seamless)
then run `build.sh` to finish setup.

Alternatively, the paths can also be passed as command line arguments, for example,

```
cmake -DPYTHON_INCLUDE_DIRS=/Users/xywei/anaconda/include/python3.6m/ -DPYTHON_LIBRARIES=/Users/xywei/anaconda/lib/libpython3.6m.dylib -DBOOST_ROOT=/Users/xywei/opt/ -DDEAL_II_DIR=/Users/xywei/opt/ -DNUMPY_LIBRARIES=/Users/xywei/opt/lib/libboost_numpy3.dylib ..
```

## Usage

The script `build.sh` also copies the built library to voluemential path so that you can use

```
import voluemential.meshgen
```

directly.

NOTE: on MacOS X the linker does not embed full path for libboost_python as RPATH for some mysterious reason. To work around this issue, you need to add the related paths into `DYLD_LIBRARY_PATH`.
