#!/bin/bash

#SBATCH --job-name=compute_psrs
#SBATCH --output=/scratch_dgxl/as5023/ardan-S/Lunar_point_model/data_processing/logs/compute_psrs.log

#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --mem=128G
#SBATCH --time=03:00:00

source /scratch_dgxl/as5023/conda/miniconda3/etc/profile.d/conda.sh
conda activate IRP

export MPLCONFIGDIR=/scratch_dgxl/as5023/.cache/matplotlib

start_time=$(date +%s)
echo "Job started on $(date)"

# Navigate to the directory where your script is located
cd $SLURM_SUBMIT_DIR

# Run the Python script
python ../compute_psrs.py --n_workers 4

end_time=$(date +%s)

echo "Job finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"