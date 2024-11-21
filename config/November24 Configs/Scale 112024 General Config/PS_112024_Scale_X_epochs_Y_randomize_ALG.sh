#!/bin/bash                                                                     
#SBATCH --job-name=PS_112024                                                    
#SBATCH --output=out_PS_112024_%A_%a.txt  # %A is the job array ID, %a is the task ID
#SBATCH --error=err_PS_112024_%A_%a.txt                                             
#SBATCH -p sched_mit_sloan_gpu_r8                                               
#SBATCH --ntasks=1                                                              
#SBATCH --cpus-per-task=2                                                       
#SBATCH --mem-per-cpu=64GB                                                   
#SBATCH --time=24:00:00                                                         
#SBATCH --gres=gpu:1  
#SBATCH --array=0-4      

source /etc/profile.d/modules.sh                                                
module load sloan/python/3.11.4

# Accept scale and epochs as command-line arguments
scale=$1
epochs=$2
alg=$3

# Check if arguments are provided
if [ -z "$scale" ] || [ -z "$epochs" ] || [ -z "$alg" ]; then
  echo "Usage: sbatch $0 <scale> <epochs> <alg>"
  exit 1
fi

echo "Scale: $scale, Epochs: $epochs, Algorithm: $alg"

# Pass scale, epochs, and task ID to the Python script
python3 PS_112024_Scale_X_epochs_Y_randomize_ALG.py $SLURM_ARRAY_TASK_ID $scale $epochs $alg