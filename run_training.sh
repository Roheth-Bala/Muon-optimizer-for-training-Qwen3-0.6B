#!/bin/bash

#SBATCH --job-name=muon_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=17:00:00
#SBATCH --output=/tmp/%u_%j.out
#SBATCH --error=/tmp/%u_%j.err

echo "Job Started: $(date)"
echo "Running on node: $(hostname)"

# 1. Temporary working directory (fast local disk)
JOB_TMP_DIR=$(mktemp -d /tmp/${USER}job${SLURM_JOB_ID}_XXXX)
echo "Created temporary directory: $JOB_TMP_DIR"

# 2. Copy script to temp dir
cp muon_training.py $JOB_TMP_DIR/
cd $JOB_TMP_DIR
echo "Current working directory: $(pwd)"

# 3. Activate conda
source /home/e/e1415353/my_projects/miniconda3/etc/profile.d/
conda activate ml

# 4. Log directory in home
LOG_DIR=/home/e/e1415353/my_projects/Assignment/
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/muon_training_${SLURM_JOB_ID}.log

# 5. Run training with live logging
echo "muon_training.py..."
python muon_training.py 2>&1 | tee $LOG_FILE

# 6. Also copy raw SLURM output/error
cp /tmp/${USER}${SLURM_JOB_ID}.out $LOG_DIR/muon_training${SLURM_JOB_ID}.slurm.out
cp /tmp/${USER}${SLURM_JOB_ID}.err $LOG_DIR/muon_training${SLURM_JOB_ID}.slurm.err

# 7. Cleanup
rm -rf $JOB_TMP_DIR
echo "Job Finished: $(date)"