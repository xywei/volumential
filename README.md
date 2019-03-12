# Volumential

Volumential (VOLUME poteNTIAL) provides toolset to solve volume potential integral equations
efficiently via Fast Multipole Method.

[Name `volumential` courtesy of Andreas Klöckner.](https://gitlab.tiker.net/xywei/volumential/issues/2)

This repository uses `git-lfs`.

**NOTE**: If you would like to skip downloading large resource files (IPython
notebooks etc.), use
```
GIT_LFS_SKIP_SMUDGE=1 git clone SERVER-REPOSITORY
```
This does not affect using the library.

Volumential is under the MIT license.

## Singular Integral Methods

- `Transform`: Apply coordinate transforms to remove the singularity (or, hide the singularity into the Jacobian of the transform).

- `DrosteSum`: Subdivide the box iteratively into layers of "bricks" towards the singular point. The name comes from **Droste effect** ([Wikipedia](https://en.wikipedia.org/wiki/Droste_effect)).

## TODOs

- [x] PEP8 compliant for `volumential/`
- [ ] PEP8 compliant for all subdirectories.
- [ ] Add type hints.
- [ ] Improve docstrings.

## Notes on meshgen_dealii

A simple way to compile `meshgen_dealii`:

- Install everything in `conda-forge`
- Download `deal-ii` and compile it with all things disabled (~15min)
- Build `meshgen11_dealii` under `contrib` in the conda env (`python 3.6` must be used)
