#!/bin/sh
#
#SBATCH --account=edu            # The account name for the job.
#SBATCH --job-name=Add2          # The job name.
#SBATCH --gres=gpu:4             # Request 4 gpu (Up to 4 on K80s, or up to 2 on P100s are valid).
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=1:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.
 
module load cuda11.2/toolkit
./vecadd00 500
 
# End of script
