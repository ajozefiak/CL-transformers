#!/bin/bash                                                                     
#SBATCH --job-name=PS_100724                                                    
#SBATCH --output=out_PS_100724_%A_%a.txt  # %A is the job array ID, %a is the task ID
#SBATCH --error=err_PS_100724_%A_%a.txt                                             
#SBATCH -p sched_mit_sloan_gpu_r8                                               
#SBATCH --ntasks=1                                                              
#SBATCH --cpus-per-task=2                                                       
#SBATCH --mem-per-cpu=32GB                                                      
#SBATCH --time=24:00:00                                                         
#SBATCH --gres=gpu:1  
#SBATCH --array=10-25                                                         

source /etc/profile.d/modules.sh                                                
module load sloan/python/3.11.4

python3 PS_100724_256_neurons_500_tasks.py $SLURM_ARRAY_TASK_ID