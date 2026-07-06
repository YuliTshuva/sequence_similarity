#!/bin/bash
#SBATCH -p medium
#SBATCH --cpu-freq=High
#SBATCH -c 20
#SBATCH --mem=2G
#SBATCH -t 100:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log

# Activate virtual environment
source ../polynomial_decomposition/venv/bin/activate

# Run your Python script
python adversarial_examples_extraction.py --dataset_prefix "six"