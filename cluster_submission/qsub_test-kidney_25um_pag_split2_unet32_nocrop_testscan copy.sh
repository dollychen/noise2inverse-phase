#   This is the most basic QSUB file needed for this cluster.
#   Further examples can be found under /share/apps/examples
#   Most software is NOT in your PATH but under /share/apps
#
#   For further info please read http://hpc.cs.ucl.ac.uk
#   For cluster help email cluster-support@cs.ucl.ac.uk
#
#   NOTE hash dollar is a scheduler directive not a comment.


# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=120G
#$ -l h_rt=1:00:00
#$ -l gpu=true,gpu_type=h100


#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N test_kidney_25um_pag_split2_unet64_nocrop_testscans
#$ -wd /home/daichen/code/noise2inverse-phase

#The code you want to run now goes here.

nvidia-smi

hostname
date

# source /share/apps/source_files/anaconda/conda-2022-5.source
source /share/apps/source_files/python/python-3.11.9.source
source /home/daichen/p_envs/noise2inverse_linux/bin/activate



python3 train.py --config ./configs/test/train_config_kidney.yaml

date
