#!/bin/bash

# Generate hmm_hrf_noise_<r> datasets: hmm_hrf plus white measurement noise
# Usage: sbatch --chdir=<repo> sbatch_hrf_noise.sh hmm_hrf_noise_0.25 hmm_hrf_noise_1.0 hmm_hrf_noise_4.0

#SBATCH -J hrf_noise
#SBATCH -A win.prj
#SBATCH -p short
#SBATCH --output=./log/slurm_hrf_noise_%j.out
#SBATCH -c 1
#SBATCH --mem-per-cpu=16G

echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Run on host: "`hostname`
echo "Started at: "`date`

module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate osld

python examples/simulation/evaluation_paper.py "$@" || exit 1

# Fail (so afterok training arrays never start) unless each dataset has exactly 500 files the data loader would read
for name in "$@"; do
    n=$(ls ./data/node_timeseries/simulation_final/$name | grep -cE '\.(txt|npy|mat|fif)$')
    echo "$name: $n loadable files"
    [ "$n" -eq 500 ] || exit 1
done
echo "Finished at: "`date`
