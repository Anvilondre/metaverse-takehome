#!/bin/bash
#SBATCH -p rtx6000
#SBATCH --qos normal
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --job-name=bert_multilingual
#SBATCH --output=slurm.log
#SBATCH --ntasks=1
#SBATCH --time=03:00:00

date;hostname;pwd

python3 BERT_Multilingual.py