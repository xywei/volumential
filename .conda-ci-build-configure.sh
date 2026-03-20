if [ "$(uname)" = "Darwin" ]; then
  export CPATH="${CONDA_PREFIX}/include${CPATH:+:${CPATH}}"
  export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export CPPFLAGS="${CPPFLAGS:-} -I${CONDA_PREFIX}/include"
  export CFLAGS="${CFLAGS:-} -I${CONDA_PREFIX}/include"
  export LDFLAGS="${LDFLAGS:-} -L${CONDA_PREFIX}/lib"
fi
