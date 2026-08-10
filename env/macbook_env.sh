# ---------------------------------------------------------------------------
# GRACE build/run environment for macOS Apple Silicon (Homebrew toolchain).
#
#   source env/mac-homebrew.sh
#
# Provides the compiler, dependency, and runtime settings the Mac build
# needs, then prints a ready-to-paste cmake configure line.  Verified on an
# Apple M-series with Homebrew LLVM + open-mpi 5.x + Homebrew Kokkos 5.1.1.
#
# Two macOS-only source fixes are already in-tree (allocator alignment clamp
# in include/grace/utils/allocators.hh; keg-libomp pin in
# cmake/setup_kokkos.cmake); nothing to do here for them.
# ---------------------------------------------------------------------------

# --- Toolchain: Homebrew LLVM clang (NOT Apple clang; needs OpenMP) ---------
export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX="/opt/homebrew/opt/llvm/bin/clang++"

# --- MPI (Homebrew open-mpi) ------------------------------------------------
export MPI_HOME="/opt/homebrew/opt/open-mpi"
export PATH="${MPI_HOME}/bin:/opt/homebrew/opt/llvm/bin:${PATH}"

# --- Dependency roots -------------------------------------------------------
# Kokkos + Kokkos-Kernels.  Kokkos comes from Homebrew; Kokkos-Kernels is a
# local source build matching the Kokkos version (see the note below).
export Kokkos_ROOT="/opt/homebrew"
export KokkosKernels_ROOT="${HOME}/Codes/Libs/kokkos-kernels-install"
# p4est/sc are a local install (not in Homebrew).
export P4EST_ROOT="${HOME}/Codes/Libs/p4est-install"
# HDF5 with MPI support (keg-only in Homebrew).
export HDF5_ROOT="/opt/homebrew/opt/hdf5-mpi"

# --- OpenMP runtime ---------------------------------------------------------
# Apple M-series: pin to the performance cores only (adjust P-core count to
# your chip: M1/M2/M3 Pro/Max = 6-12, base M-series = 4).  Including the
# efficiency cores stalls the OpenMP join barrier.  macOS has no real thread
# pinning, so OMP_PROC_BIND is a no-op (leave it false to silence warnings).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PROC_BIND=false

echo "GRACE macOS environment set:"
echo "  CXX               = ${CXX}"
echo "  Kokkos_ROOT        = ${Kokkos_ROOT}"
echo "  KokkosKernels_ROOT = ${KokkosKernels_ROOT}"
echo "  P4EST_ROOT         = ${P4EST_ROOT}"
echo "  OMP_NUM_THREADS    = ${OMP_NUM_THREADS} (P-cores; PROC_BIND off)"
echo
echo "Configure (must be Release: empty CMAKE_BUILD_TYPE -> -O0 -> ~40x slower):"
cat <<'EOF'
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
    -DGRACE_ENABLE_OMP=ON -DGRACE_METRIC_EVOL=Z4 -DGRACE_CARTESIAN_COORDINATES=ON \
    -Dp4est_INCLUDE_DIR=$P4EST_ROOT/include \
    -Dp4est_LIBRARY=$P4EST_ROOT/lib/libp4est.a \
    -Dsc_LIBRARY=$P4EST_ROOT/lib/libsc.a
  # add for M1 neutrino transport:
  #   -DGRACE_ENABLE_M1=ON -DGRACE_M1_NU_SPECIES=5 -DGRACE_M1_PHOTONS=ON
  # add for the ML inference module (needs KokkosKernels_ROOT above):
  #   -DGRACE_ENABLE_ML=ON

  cmake --build build -j$(sysctl -n hw.ncpu)

# Kokkos-Kernels is NOT in Homebrew.  Build once, matching the Homebrew
# Kokkos version (BLAS+BATCHED components suffice for the ML module):
#   git clone --branch <kokkos-version> https://github.com/kokkos/kokkos-kernels.git
#   cmake -S kokkos-kernels -B kk-build -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
#     -DKokkos_ROOT=/opt/homebrew \
#     -DKokkosKernels_ENABLE_ALL_COMPONENTS=OFF \
#     -DKokkosKernels_ENABLE_COMPONENT_BLAS=ON \
#     -DKokkosKernels_ENABLE_COMPONENT_BATCHED=ON \
#     -DKokkosKernels_ADD_DEFAULT_ETI=OFF \
#     -DCMAKE_INSTALL_PREFIX=$HOME/Codes/Libs/kokkos-kernels-install
#   cmake --build kk-build -j && cmake --install kk-build
EOF

