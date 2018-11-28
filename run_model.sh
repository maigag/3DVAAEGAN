# BASH SCRIPT TO RUN MODEL ON LEONHARD #
# run this on script directory #

# Set up remote project path
REMOTEFOLDER="/cluster/scratch/$USER/THESIS"

# Data paths
DATAPATH="$REMOTEFOLDER/SUNCG"
TRAINPATH="$DATAPATH/Scene"
XPATH="$TRAINPATH/gt/"
XPATH2="./Data"
YPATH="$TRAINPATH/dc/"
TESTPATH="$DATAPATH/Data_Test"
LABELPATH="$TRAINPATH/labels.txt"
OUTPUTPATH="$DATAPATH/OUTPUT"
SAMPLEOUTPATH="$TRAINPATH/samples"

# Model parameters

# RUN ON LEONHARD
echo "SUBMIT JOB ON LEONHARD . . ."
module load python_gpu/3.6.4
bsub -n 4 -W 120:00 -R "rusage[mem=4500, ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]"  python main.py --X_path=$XPATH --Y_path=$YPATH --label_map_path=$LABELPATH --sample_output_path=$SAMPLEOUTPATH --load_model=49
echo "DONE!"
