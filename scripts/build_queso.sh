#!/usr/bin/env bash
# Build official Dakota 6.23 source with bundled QUESO, DREAM and native GP.
# Set SOURCE, BUILD, PREFIX, and optionally TOOLCHAIN and JOBS before running.
set -euo pipefail
: "${SOURCE:?Set SOURCE to the unpacked official Dakota 6.23 source directory}"
: "${BUILD:?Set BUILD to an out-of-tree build directory}"
: "${PREFIX:?Set PREFIX to an installation directory}"
: "${TOOLCHAIN:?Set TOOLCHAIN to a prefix containing CMake, compilers, GSL, Boost, BLAS/LAPACK}"
JOBS="${JOBS:-4}"
export PATH="$TOOLCHAIN/bin:$PATH"
export CC="${CC:-$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++}"
export FC="${FC:-$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gfortran}"
"$TOOLCHAIN/bin/cmake" -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_PREFIX_PATH="$TOOLCHAIN" -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release -DDAKOTA_ENABLE_TESTS=OFF -DBUILD_TESTING=OFF \
  -DHAVE_QUESO=ON -DDAKOTA_HAVE_GSL=ON -DHAVE_HOPSPACK=ON \
  -DHAVE_QUESO_GPMSA=OFF -DDAKOTA_HAVE_MPI=OFF \
  -DBLAS_LIBS="$TOOLCHAIN/lib/libopenblas.so" \
  -DLAPACK_LIBS="$TOOLCHAIN/lib/libopenblas.so" \
  -DCMAKE_EXE_LINKER_FLAGS="-lm -Wl,-rpath-link,$TOOLCHAIN/lib" -DDL_LIBRARY=dl \
  -DDAKOTA_NO_FIND_TRILINOS=ON \
  -DHAVE_DDACE=OFF -DHAVE_ROL=OFF -DHAVE_JEGA=OFF \
  -DHAVE_NOMAD=OFF -DHAVE_PSUADE=OFF \
  -DHAVE_NLPQL=OFF -DHAVE_NL2SOL=OFF -DHAVE_NPSOL=OFF
# Dakota installs auxiliary executables such as lhsdrv even when only dakota
# was requested; build the default target before installing the full tree.
"$TOOLCHAIN/bin/cmake" --build "$BUILD" -j "$JOBS"
"$TOOLCHAIN/bin/cmake" --install "$BUILD"
LD_LIBRARY_PATH="$TOOLCHAIN/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PREFIX/bin/dakota" -v
