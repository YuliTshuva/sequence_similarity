#!/bin/bash
#SBATCH -p killable
#SBATCH --cpu-freq=High
#SBATCH -c 75
#SBATCH --mem=2G
#SBATCH -t 24:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log

# Activate virtual environment
source venv/bin/activate

# Run your Python script
python your_script.py