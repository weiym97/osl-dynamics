#!/bin/bash
# Run one model selection job per array task.
#
# Usage:
#
#     osl-dynamics-evaluate config.yml --init   # prints the number of jobs, N
#     sbatch --array=1-N submit_slurm.sh config.yml
#
# When every task has finished:
#
#     osl-dynamics-evaluate config.yml --summarise
#
# Change the resources, partition and environment for your cluster.

#SBATCH --job-name=osld-evaluate
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

source activate osld

osl-dynamics-evaluate "$1" --index "$SLURM_ARRAY_TASK_ID"
