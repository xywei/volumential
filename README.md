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

## Notes on meshgen_dealii

A simple way to compile `meshgen_dealii`:

- Install everything in `conda-forge`
- Download `deal-ii` and compile it with all things disabled (~15min)
- Build `meshgen11_dealii` under `contrib` in the conda env (`python 3.6` must be used)

## License

Copyright © `2019` `Xiaoyu Wei, Andreas Klöckner`

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgement

We would like to thank people and organizations that contributed to or supported the
Volumential project. See [ACKNOWLEDGEMENTS](./ACKNOWLEDGEMENTS.md) for details.
