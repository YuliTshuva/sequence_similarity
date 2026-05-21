#!/bin/bash
#SBATCH -p short
#SBATCH --cpu-freq=High
#SBATCH -c 75
#SBATCH --mem=2G
#SBATCH -t 24:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log

# Activate virtual environment
source ../polynomial_decomposition/venv/bin/activate

# Run your Python script
python adversarial_examples_extraction.py