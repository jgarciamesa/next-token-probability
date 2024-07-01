#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 2            # number of cores 
#SBATCH -t 0-01:00:00   # time in d-hh:mm:ss
#SBATCH -p htc          # partition 
#SBATCH -q public       # QOS
#SBATCH --gres=gpu:a100:2
#SBATCH -Ca100_80
#SBATCH --mem=24GB
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load required software
module load mamba/latest

#Activate our environment
source activate genai23.10

python scripts/patent_classification_llm.py data/patent_classification_data.csv Llama3-8b-instruct
