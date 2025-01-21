#!/bin/bash                                                                     
#SBATCH --job-name=SL_121224                                                    
#SBATCH --output=out_SL_121224_%A_%a.txt  # %A is the job array ID, %a is the task ID
#SBATCH --error=err_SL_121224_%A_%a.txt                                             
#SBATCH -p sched_mit_sloan_gpu_r8                                               
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00                                                         
#SBATCH --gres=gpu:1  
#SBATCH --array=0   

source /etc/profile.d/modules.sh                                                
module load sloan/python/3.11.4

# Accept scale and epochs as command-line arguments
n_embd=$1
n_head=$2
n_layer=$3
D=$4
alg=$5
lr=$6
name=$7
log_freq=$8

# Check if arguments are provided
if [ -z "$n_embd" ] || [ -z "$n_head" ] || [ -z "$n_layer" ] || [ -z "$alg" ] || [ -z "$lr" ] || [ -z "$D" ] || [ -z "$name" ]; then
  echo "Usage: sbatch $0 <scale> <epochs> <alg>"
  exit 1
fi

echo "n_embd: $n_embd, n_head: $n_head, n_layer: $n_layer, D: $D, alg: $alg, lr: $lr, name: $name, log_freq: $log_freq"

# Pass scale, epochs, and task ID to the Python script
python3 scaling_laws.py $SLURM_ARRAY_TASK_ID $n_embd $n_head $n_layer $D $alg $lr $name $log_freq