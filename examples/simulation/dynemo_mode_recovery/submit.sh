#!/bin/bash
# Run one line of a jobs file per array task: "<run name> [run.py args]".
# Usage: sbatch -a 1-N submit.sh <jobs_file> <tag>
#SBATCH -J recovery
#SBATCH -A win.prj
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --output=/gpfs3/well/win-fmrib-analysis/users/uap971/osl-dynamics/examples/simulation/dynemo_mode_recovery/output/logs/%A_%a.out

HERE=/gpfs3/well/win-fmrib-analysis/users/uap971/osl-dynamics/examples/simulation/dynemo_mode_recovery
read -r NAME ARGS <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
TAG=${2:-main}

module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate osld

cd "$HERE/output"
python "$HERE/run.py" "$HERE/output/runs/$TAG/$NAME" $ARGS
