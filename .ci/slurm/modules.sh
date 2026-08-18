#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export LD_PRELOAD=${LD_PRELOAD:-}

source /etc/profile.d/lmod.sh
module purge
module load cmake/3.31.6
module load openmpi/5.0.10-nvhpc26.3-gnu-cuda12-auto
module load apptainer/1.4.4
