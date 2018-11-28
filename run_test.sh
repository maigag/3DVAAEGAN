# BASH SCRIPT TO RUN MODEL ON LEONHARD #
# run this on script directory #

# Set up remote project path
REMOTEFOLDER="/cluster/scratch/$USER/THESIS"

# Data paths
DATAPATH="$REMOTEFOLDER/SUNCG"
TESTPATH="$DATAPATH/Scene_Test"

# Model parameters

# RUN ON LEONHARD
echo "SUBMIT JOB ON LEONHARD . . ."
module load python_gpu/3.6.4
bsub -n 1 -W 8:00 -R "rusage[mem=250000, ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]" python evaluate.py --test_folder=$TESTPATH
echo "DONE!"
