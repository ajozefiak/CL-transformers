#!/bin/bash                                                                     
#SBATCH --job-name=PS_120124                                                    
#SBATCH --output=out_PS_120124_%A_%a.txt  # %A is the job array ID, %a is the task ID
#SBATCH --error=err_PS_120124_%A_%a.txt                                             
#SBATCH -p sched_mit_sloan_gpu_r8                                               
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=10
#SBATCH --mem=128GB
#SBATCH --time=24:00:00                                                         
#SBATCH --gres=gpu:1  
#SBATCH --array=0-4      

source /etc/profile.d/modules.sh                                                
module load sloan/python/3.11.4

# Accept scale and epochs as command-line arguments
scale=$1
epochs=$2
tasks=$3
alg=$4
lr=$5

# Check if arguments are provided
if [ -z "$scale" ] || [ -z "$epochs" ] || [ -z "$tasks" ] || [ -z "$alg" ] || [ -z "$lr" ]; then
  echo "Usage: sbatch $0 <scale> <epochs> <alg>"
  exit 1
fi

echo "Scale: $scale, Epochs: $epochs, Algorithm: $alg, lr: $lr"

# Pass scale, epochs, and task ID to the Python script
python3 PS_120124_Scale_X_epochs_Y_tasks_Z_ALG_lr_LR.py $SLURM_ARRAY_TASK_ID $scale $epochs $tasks $alg $lr