#!/bin/bash

#SBATCH --job-name=download_data          # Job name
#SBATCH --output=/home/as5023/ardan-S/Lunar_point_model/process_data/logs/download_data.log       # Output log
#SBATCH --partition=dgxl_irp              # Partition name (dedicated partition)
#SBATCH --qos=dgxl_irp_high               # QoS, use high priority to access GPU faster
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --ntasks=1                        # Run on a single task
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=16G                         # Memory limit per node
#SBATCH --time=00:03:00                   # Time limit

# Load necessary modules
module load python/3.8

# Navigate to the directory where your script is located
cd $cd $SLURM_SUBMIT_DIR

# Activate your conda/venv environment
source /scratch_dgxl/as5023/myenv/bin/activate

# Run the Python script
python download_data.py

# Print completion message
echo "Job finished successfully"
