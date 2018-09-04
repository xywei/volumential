# Mesh generation via Deal.II

`meshgen11` generates a set of quadrature points and weights using `Deal.II`.
Different from the other `meshgen`, it uses `pybind11` instead of `boost::python`.

## Build

To build `meshgen11`, you need:

- Python 3.6
- Deal.II 8.4.2 or later

## Usage

The script `build.sh` also copies the built library to voluemential path so that you can use

```
import voluemential.meshgen
```

directly.
