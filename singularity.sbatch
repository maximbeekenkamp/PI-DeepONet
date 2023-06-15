#!/bin/bash
#SBATCH -t 50:00:00
#SBATCH --mem=128g
##SBATCH -p 3090-gcondo
#SBATCH -p a6000-gcondo
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o job_tf_%j.out
#SBATCH -e job_tf_%j.err

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

export SINGULARITY_BINDPATH="/gpfs/scratch,/gpfs/data"

#singularity exec --nv /gpfs/data/gk/sgoswam5/tf2.simg python -u train_AE.py
singularity exec --nv /gpfs/data/gk/sgoswam5/tf2.simg python -u main.py
