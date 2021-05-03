#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 71:59:00
#SBATCH --mem=12000
#SBATCH --gres=gpu:v100:1
#SBATCH -J dl2
#SBATCH -o dl2.out.%j
#SBATCH -e dl2.err.%j
#SBATCH --account=project_2002605
#SBATCH

module purge
module load pytorch/1.7

python bert_data_loader_real_data_strat.py
