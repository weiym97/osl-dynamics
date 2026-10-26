#!/bin/bash 

# Specify a job name 
#SBATCH -J nfindr

# Account name and target partition 
#SBATCH -A win.prj 
#SBATCH -p short

# Check whether the directory log is available. Create otherwise.
#if [ ! -d "log" ]; then
#    mkdir log
#fi

# Generate timestamp
#timestamp=$(date +"%Y%m%d_%H%M%S")

# Log locations which are relative to the current 
# working directory of the submission   
#SBATCH --output=./log/slurm_%A_%a.out
 

# Parallel environment settings 
#  For more information on these please see the documentation 
#  Allowed parameters: 
#   -c, --cpus-per-task 
#   -N, --nodes 
#   -n, --ntasks 
#SBATCH -c 1 
#SBATCH --mem-per-cpu=50G

# Some useful data about the job to help with debugging 
echo "------------------------------------------------" 
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID" 
echo "Run on host: "`hostname` 
echo "Operating system: "`uname -s` 
echo "Username: "`whoami` 
echo "Started at: "`date` 
echo "------------------------------------------------" 

# Load the environment
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate osld # Environment name

# Begin writing your script here 

#python post_analysis.py $SLURM_ARRAY_TASK_ID
#python main_copy.py $SLURM_ARRAY_TASK_ID split 1 1
#ulimit -n 20480
# Config path is the first argument, e.g. sbatch -a 1-64 sbatch_nfindr.sh ./results_final/simulation/dynemo_fair_nfindr/config_simulation.yaml
python evaluate_main.py $SLURM_ARRAY_TASK_ID "$1"
# End of job script
