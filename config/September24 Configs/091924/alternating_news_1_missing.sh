#!/bin/bash                                                                     
#SBATCH --job-name=alt_news_1                                                 
#SBATCH --output=out_alt_news_1_%A_%a.txt  # %A is the job array ID, %a is the task ID
#SBATCH --error=err_alt_news_1_%A_%a.txt                                             
#SBATCH -p sched_mit_sloan_gpu_r8                                               
#SBATCH --ntasks=1                                                              
#SBATCH --cpus-per-task=1                                                       
#SBATCH --mem-per-cpu=32GB                                                      
#SBATCH --time=4:00:00                                                         
#SBATCH --gres=gpu:1  
#SBATCH --array=1-3                                                          

source /etc/profile.d/modules.sh                                                
module load sloan/python/3.11.4

python3 alternating_news_1_missing.py $SLURM_ARRAY_TASK_ID