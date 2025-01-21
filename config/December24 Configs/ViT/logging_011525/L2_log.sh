#!/bin/bash                                                                     
#SBATCH --job-name=CI_lr_sweep                                                    
#SBATCH --output=out_CI_lr_sweep_%A_%a.txt  # %A is the job array ID, %a is the task ID
#SBATCH --error=err_CI_lr_sweep_%A_%a.txt                                             
#SBATCH -p sched_mit_sloan_gpu_r8                                               
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=24:00:00                                                         
#SBATCH --gres=gpu:1  
#SBATCH --array=0      

source /etc/profile.d/modules.sh                                                
module load sloan/python/3.11.4

# Accept scale and epochs as command-line arguments
layers=$1
lr=$2

echo "Scale: $layers, lr: $lr"

python3 L2_log.py $SLURM_ARRAY_TASK_ID $layers $lr