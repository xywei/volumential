#!/bin/bash

mkdir -p build
cd build
cmake ..
make
cd ..

echo Module built:
echo ${PWD}/build/meshgen.so

if [ -f build/meshgen.so ]; then
  cp build/meshgen.so ../../volumential/meshgen.so
elif [ -f build/meshgen.dylib ]; then
  cp build/meshgen.dylib ../../volumential/meshgen.so
else
  echo "Something went wrong. Build failed."
  exit 1
fi

echo Library copied to volumential path.
