# Volumential

Volumential (VOLUME poteNTIAL) provides toolset to solve volume potential integral equations
efficiently via Fast Multipole Method.

[Name `volumential` courtesy of Andreas Klöckner.](https://gitlab.tiker.net/xywei/volumential/issues/2)

Volumential is under the MIT license.

## Singular Integral Methods

- `Transform`: Apply coordinate transforms to remove the singularity (or, hide the singularity into the Jacobian of the transform).

- `DrosteSum`: Subdivide the box iteratively into layers of "bricks" towards the singular point. The name comes from **Droste effect** ([Wikipedia](https://en.wikipedia.org/wiki/Droste_effect)).
