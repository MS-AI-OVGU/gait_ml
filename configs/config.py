import os
from pathlib import Path

# SET STATIC PATHS
ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT_DIR = ROOT_DIR / 'gait_ml'

DATA_DIR = ROOT_DIR / 'data'
DATASETS_01 = DATA_DIR / 'dataset_01' /'all_data'
ENDO_DATA_DIR = DATASETS_01 / 'Enode'
SMART_PHONE_DATA_DIR = DATASETS_01 / 'Smartphone'

DATASETS_02 = DATA_DIR / 'dataset_02' / '21072025'
DATASETS_02_T1 = DATASETS_02 / 'Termin 1 Vicon'
DATASETS_02_T2 = DATASETS_02 / 'Termin 2 Vicon'

DATASETS = PROJECT_ROOT_DIR / 'datasets'
DATAFRAMES = DATASETS / 'dataframes'
DATA_PROCESSED = DATASETS / 'processed'
VISUALIZE_DIR = PROJECT_ROOT_DIR / 'visualize'
OUTPUT_DIR = PROJECT_ROOT_DIR / 'output'


# SET TRAINING- DATA PARAMETERS
sensor_cols = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z']
SENSOR_COLS  = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z']

TRAIN_PATH   = os.path.join(DATA_PROCESSED,"event_classification","endo_smartphone_2mW_train_data.csv") #ENODE labels
TEST_PATH    = os.path.join(DATA_PROCESSED,"event_classification","endo_smartphone_2mW_test_data.csv") #ENODE labels

train_val_subjects = ['t14_2mW','s4_2mW_1','s6_2mW_1','s4_2mW_2','s6_2mW_2']
test_subjects  = ['t10_2mW','s2_2mW_1','t11_2mW']

train_csv = TRAIN_PATH
test_csv  = TEST_PATH

train_dir = DATA_PROCESSED / "seq2seq" / "train"
test_dir = DATA_PROCESSED / "seq2seq" / "test"
val_dir = DATA_PROCESSED / "seq2seq" / "val"

SEED         = 31101995
fs = 100  # Sampling frequency in Hz

# SET MODEL SAVING PARAMETERS

EXPERIMENT_NAME = "seq2seq_hid64_bidir"
EXPERIMENT_DIR = OUTPUT_DIR / "learning" / EXPERIMENT_NAME
LOG_DIR        = EXPERIMENT_DIR / "logs"
CKPT_DIR       = EXPERIMENT_DIR / "ckpts"
log_dir        = EXPERIMENT_DIR / "logs"
ckpt_dir       = EXPERIMENT_DIR / "ckpts"
plots_dir      = EXPERIMENT_DIR / "plots"
results_dir   = EXPERIMENT_DIR / "results"

# SET TRAINING- MODEL PARAMETERS
n_classes = 3
input_dim = 6

earlystop_patience = 100
earlystop_delta = 0.0005
dropout = 0.3
grad_clip = 1.0
# a. LR scheduler: StepLR
step_size = 30
gamma = 0.1
# b. LR scheduler: ReduceLROnPlateau
scheduler_patience = 5
min_lr = 0.0


#SET TEST and TASK input parameters from app
TEST = "gait"
TASK = '2min'
subject = 't11_2mW'

# subject
#TEST = "sts"
#TASK = "5rep"

#TEST = "tug"
#TASK = "single"
#TASK = "dual"