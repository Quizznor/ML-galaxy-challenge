#!/bin/bash

#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=6        # the number of CPU cores per node
#SBATCH --mem=32G                   # how much memory is needed per node (units can be: K, M, G, T)
#SBATCH --gres gpu:1
#SBATCH --partition=gpu3090        # on which partition to submit the job
#SBATCH --time=2-00:00:00             # the max wallclock time (time limit your job will run)

#SBATCH --job-name=Train_VAE         # the name of your job
#SBATCH --output=output.dat         # the file where output is written to (stdout & stderr)
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=h_schu55@uni-muenster.de # your mail address

# LOAD MODULES HERE IF REQUIRED

ml palma/2020b
ml fosscuda
ml TensorFlow
ml matplotlib
ml Pillow
ml jax

# START THE APPLICATION
cd /scratch/tmp/h_schu55/galaxies/ML-galaxy-challenge/

python autoencoder/training.py --ConfigFile config.json