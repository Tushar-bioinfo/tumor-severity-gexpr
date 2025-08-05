#!/bin/bash
#SBATCH --job-name=optuna-cnn
#SBATCH --output=slurm/optuna-cnn.out
#SBATCH --error=slurm/optuna-cnn.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com

module load nextflow
nextflow run main.nf -profile slurm
