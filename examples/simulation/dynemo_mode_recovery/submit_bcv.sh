#!/bin/bash
# Run one line of a jobs file per array task: "[bcv_oracle.py args]" (run from the repository root).
# Usage: sbatch -a 1-N submit_bcv.sh <jobs_file> <tag>
#SBATCH -J bcvoracle
#SBATCH -A win.prj
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --output=/gpfs3/well/win-fmrib-analysis/users/uap971/osl-dynamics/examples/simulation/dynemo_mode_recovery/output/logs/%A_%a.out

REPO=/gpfs3/well/win-fmrib-analysis/users/uap971/osl-dynamics
HERE=$REPO/examples/simulation/dynemo_mode_recovery
ARGS="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
TAG=${2:-bcv}

module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate osld

cd "$REPO"
python "$HERE/bcv_oracle.py" "$HERE/output/runs/$TAG" $ARGS
