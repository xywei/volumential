#!/bin/bash
## ---------------------------------------------------------------------
##
## Copyright (C) 2017 - 2018 Xiaoyu Wei
##
## This file is part of Volumential.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.
##
## ---------------------------------------------------------------------

# This scripe updates the copyright year of every file based on the last
# modification recorded in the git logs

if test ! -d volumential -o ! -d test -o ! -d examples ! -d contrib ; then
  echo "Error: This script must be run from the top-level directory of Volumential."
  exit
fi

files="`echo volumential/*py \
             test/*py \
             contrib/*py \
             examples/*py`
       `find . -not -path '*pybind11*' -name CMakeLists.txt`
       `find . -not -path '*pybind11*' -name Makefile`
       `find . -not -path '*pybind11*' | egrep '\.(sh)$'`
       "

for i in $files ; do
  last_year=`git log -n 5 --date=short --format="format:%cd %s" $i | \
             egrep -i -v "Update copyright dates" | \
             head -n 1 | \
             perl -p -e 's/^(\d\d\d\d)-.*/\1/g;'`

  # The first year may differ from the git log if the file was copied
  # from elsewhere
  first_year=`cat $i | egrep 'Copyright \(C\) [0-9]{4}' | \
              perl -p -e "s/.*Copyright \(C\) (\d{4}).*/\1/g;"`

  echo "Processing $i: ${first_year} - ${last_year}"
  if test ! "${first_year}" = "${last_year}" ; then
    perl -pi -e "s/(Copyright \(C\) \d{4})( - \d{4})?(, \d{4}( - \d{4})?)*/\1 - ${last_year}/g;" $i
  fi
done

echo "***************************************************************"
echo "Update finished. Please commit changes with the commit message:"
echo "    Update copyright dates"
echo "***************************************************************"

