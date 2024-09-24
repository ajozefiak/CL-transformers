#!/bin/bash                                                                     
#SBATCH --job-name=test_save                                                  
#SBATCH --output=out_test_save_%A_%a.txt  # %A is the job array ID, %a is the task ID
#SBATCH --error=err_test_save_%A_%a.txt                                             
#SBATCH -p sched_mit_sloan_batch_r8                                               
#SBATCH --ntasks=1                                                              
#SBATCH --cpus-per-task=1                                                    
#SBATCH --mem-per-cpu=8GB                                                      
#SBATCH --time=20:00                                                         
#SBATCH --gres=gpu:0  
#SBATCH --array=1                                                          

source /etc/profile.d/modules.sh                                                
module load sloan/python/3.11.4

python3 process_results_ATN_2.py