#!/bin/bash

#SBATCH --job-name=download_data          # Job name
#SBATCH --output=/scratch_dgxl/as5023/ardan-S/Lunar_point_model/data_processing/logs/download_data.log
#SBATCH --partition=dgxl_irp              # Partition name (dedicated partition)
#SBATCH --qos=dgxl_irp_high               # QoS, use high priority to access GPU faster
##SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --ntasks=1                        # Run on a single task
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --mem=16G                         # Memory limit per node
#SBATCH --time=10:00:00                   # Time limit

source /scratch_dgxl/as5023/conda/miniconda3/etc/profile.d/conda.sh
conda activate IRP

start_time=$(date +%s)

# Navigate to the directory where your script is located
cd $SLURM_SUBMIT_DIR

# Run the Python script
python ../download_data.py

end_time=$(date +%s)

echo "Job finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"