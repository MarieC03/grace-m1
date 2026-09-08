#!/bin/bash
#PBS -N lbm
#PBS -l select=1:node_type=mi300a
#PBS -l walltime=02:00:00
#PBS -o lbm.log
#PBS -j oe
#PBS -M miler@itp.uni-frankfurt.de
#PBS -m abe
#
# LBM radiation: regression gate plus the cluster-resolution runs, one MI300A node.
#
# The Cowling runs (flat and static-metric problems) come from BUILD_COWLING, the Z4c
# ones from BUILD_Z4; configure them as the headers of the parfiles state.  Set
# WHICH=cowling|z4|all and RUN_CTEST=0|1 in the environment to select what runs.
#
#   qsub examples/lbm/submit_lbm_hunter.sh
#   qsub -v WHICH=z4,RUN_CTEST=0 examples/lbm/submit_lbm_hunter.sh

source /zhome/projects/groups/xfp44203/common/grace-env-2026.sh
export HSA_XNACK=1                 # MI300A unified memory; device kernels fault without it
export OMP_NUM_THREADS=24          # 24 cores per APU
export OMP_PLACES=cores
export OMP_PROC_BIND=close

SRC=/lustre/hpe/ws13/ws13.a/ws/xfpmiler-BHNS/grace-lbm
BUILD_COWLING=${SRC}/build-lbm
BUILD_Z4=${SRC}/build-lbm-z4
RUN_ROOT=/lustre/hpe/ws13/ws13.a/ws/xfpmiler-BHNS/lbm-runs/${PBS_JOBID:-interactive}

WHICH=${WHICH:-all}
RUN_CTEST=${RUN_CTEST:-1}
# RUNS overrides WHICH: a space-separated list of parfile names to run, e.g.
#   qsub -v RUNS=lbm_curved_beam_hr,RUN_CTEST=0 examples/lbm/submit_lbm_hunter.sh
# Names containing z4 use BUILD_Z4, the rest BUILD_COWLING.
RUNS=${RUNS:-}

# One rank per APU: 4 chips, cores 0-23/24-47/48-71/72-95, GPUs 0-3.
launch () {   # launch <build> <parfile-name>
    local build=$1 name=$2
    mkdir -p "${RUN_ROOT}/${name}" && cd "${RUN_ROOT}/${name}" || exit 1
    echo "=== ${name}  ($(date +%H:%M:%S))"
    mpiexec -n 4 -ppn 4 \
        --cpu-bind list:0-23:24-47:48-71:72-95 \
        --gpu-bind list:0:1:2:3 \
        "${build}/grace" --grace-parfile "${SRC}/examples/lbm/${name}.yaml"
}

# Regression gate.  The unit tests abort in MPI_Init unless GPU-aware MPI is off: Cray
# MPICH wants libmpi_gtl_hsa linked, which only the grace binary gets (through HDF5).
if [ "${RUN_CTEST}" = "1" ]; then
    for b in "${BUILD_COWLING}" "${BUILD_Z4}"; do
        [ -d "${b}" ] || continue
        echo "=== ctest -L lbm in ${b}"
        ( cd "${b}" && MPICH_GPU_SUPPORT_ENABLED=0 ctest -L lbm --output-on-failure ) \
            || { echo "REGRESSION FAILED in ${b} -- stopping before the big runs"; exit 1; }
    done
fi

if [ -n "${RUNS}" ]; then
    for name in ${RUNS}; do
        case "${name}" in
            *z4*) launch "${BUILD_Z4}" "${name}" ;;
            *)    launch "${BUILD_COWLING}" "${name}" ;;
        esac
    done
    echo "=== done ($(date +%H:%M:%S)); output under ${RUN_ROOT}"
    exit 0
fi

if [ "${WHICH}" = "cowling" ] || [ "${WHICH}" = "all" ]; then
    launch "${BUILD_COWLING}" lbm_straight_beam_hr
    launch "${BUILD_COWLING}" lbm_crossed_beams_hr
    launch "${BUILD_COWLING}" lbm_shadow_hr
    launch "${BUILD_COWLING}" lbm_sphere_wave_ks_hr
    launch "${BUILD_COWLING}" lbm_curved_beam_hr
    launch "${BUILD_COWLING}" lbm_tov_cowling_hr
fi

if [ "${WHICH}" = "z4" ] || [ "${WHICH}" = "all" ]; then
    launch "${BUILD_Z4}" lbm_tov_z4_hr
    launch "${BUILD_Z4}" lbm_puncture_z4_hr
fi

echo "=== done ($(date +%H:%M:%S)); output under ${RUN_ROOT}"
