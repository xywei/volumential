# Design notes

Two mechanisms in Volumential are hard to read off the source, because the code
implements a derivation that lives elsewhere. These pages say what the
mechanism *is*, what guarantee it carries and where in the tree it lives. They
do not reproduce the derivations: those belong to the accompanying manuscripts,
and a second, drifting copy of them here would be worse than none.

```{toctree}
:maxdepth: 1

windowed-channels
orbit-canonicalization
```

Everything on these pages is implemented and tested in the repository; nothing
here is a plan. Where a page states a bound, it is the bound the code computes
and records, not a sharper one from a manuscript.
