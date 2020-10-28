#!/bin/sh

cd /home/firedrake/firedrake/src/
mv loo-py/ loopyy/

# under loopyy
cd loopyy/
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/loopy/loopyy/g'
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/loo.py/loo.pyy/g'
mv loopy loopyy
mv bin/loopy bin/loopyy
cd ..

# under meshmode
cd meshmode/
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/loopy/loopyy/g'
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/loo.py/loo.pyy/g'
cd ..

# under pytential
cd pytential/
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/loopy/loopyy/g'
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/loo.py/loo.pyy/g'
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/from __future__.*//g'  # to use py3.6
cd ..

# under volumential
cd volumential/
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/loopy/loopyy/g'
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/loo.py/loo.pyy/g'
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/from __future__.*//g'  # to use py3.6
cd ..

# fix egg links and make sure both loopy and loopyy are installed
rm /home/firedrake/firedrake/lib/python3.6/site-packages/loo.py.egg-link
pip install -e ./loopyy
pip install -e ./loopy
pip install -e ./loopyy
