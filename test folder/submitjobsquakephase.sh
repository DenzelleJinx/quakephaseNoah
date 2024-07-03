#!/bin/bash

#SBATCH --job-name=testNewWrapperParallelDask     # Job name
#SBATCH --output=/cluster/scratch/nmunro/Outputfile.log  # Output file
#SBATCH --error=/cluster/scratch/nmunro/Errorfile.log  # Error file
#SBATCH --time=01:00:00                   # Time limit hrs:min:sec
#SBATCH --cpus-per-task=36             # Number of CPU cores per task
#SBATCH --mem-per-cpu=2GB                # Total memory limit

# Navigate to the correct directory
cd /cluster/scratch/nmunro

# Load the required modules in the correct order
module load eth_proxy 
module load gcc/13.2.0
module load python/3.11.6

export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128

# Activate the virtual environment using the absolute path
source /cluster/scratch/nmunro/myenv/bin/activate

# Set permissions on the folder before running the Python script
chmod +r /cluster/scratch/nmunro/tpc5File

# Run your Python script from the current directory
python3 /cluster/scratch/nmunro/quakephaseNoah/Wrapper/main.py

# Deactivate the virtual environment (optional, the job ends here anyway)
deactivate

echo "Job completed."
