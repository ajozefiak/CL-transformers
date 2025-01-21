#!/bin/bash                                                                     
#SBATCH --job-name=PS_092224_256                                                    
#SBATCH --output=out_PS_092224_256_%A_%a.txt  # %A is the job array ID, %a is the task ID
#SBATCH --error=err_PS_092224_256_%A_%a.txt                                             
#SBATCH -p sched_mit_sloan_gpu_r8                                               
#SBATCH --ntasks=1                                                              
#SBATCH --cpus-per-task=2                                                       
#SBATCH --mem-per-cpu=64GB                                                      
#SBATCH --time=12:00:00                                                         
#SBATCH --gres=gpu:1  
#SBATCH --array=1-6                                                          

source /etc/profile.d/modules.sh                                                
module load sloan/python/3.11.4

python3 PS_092224_256.py $SLURM_ARRAY_TASK_ID