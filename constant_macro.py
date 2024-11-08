# epoch config
EPOCH_RANGE = 80
REFRESH_EPOCH_NUMBER = False  # Count the epoch number from 0 when fine-tuning a checkpoint
EPOCH_TRAIN_SAVE_PERIOD = 5  # Model saving period
EPOCH_TRAIN_SAVE_MIN_EPOCH = 10  # The minimum epoch to start saving the model on every epoch
EPOCH_VALIDATE_PERIOD = 5  # Model validation period
EPOCH_VALIDATE_DENSE_MIN_EPOCH = 10  # The minimum epoch to start validating the model on every epoch

# hyper params
SUB_TRAINED = None  # Specify training subnets
FT_EXCLUDED = None  # Excluded model subnets when reload a model checkpoint

# train params
BATCH_SIZE = 4  # Training batch size. Depends on your GPU memory
VALIDATE_BATCH_SIZE = 16  # Validation batch size
LEARNING_RATE = 0.0005
NEW_OPTIMIZER = False  # Whether to initiate a new optimizer when fine-tuning a model
STEP_STAIR_LENGTH = 9e10  # Number of steps to decrease the learning rate by a stair-down learning rate strategy.
# We degrade the stair-down strategy here as a normal constant learning rate.

# loader params
TRAIN_FRAMES_SIZE = 400  # Frame length of each sample during training
VALIDATE_FRAMES_SIZE = 400  # Frame length of each sample during validation
SCHEDULE_IDX = '2x'  # Training and validating targets.
# Refer the training schedule in the function "draft_schedule()" in "func.py" for more details.
EVALUATE_TERM = 'note'  # Training and validating decoding objects.
# Including "note", "dp" (damper/sustain pedal), "sp" (sostenuto pedal) and "uc" (una corda pedal).
LOAD_PORTION = 1.0  # Training dataset load portion in [0, 1]
LOAD_OFFSET = 0  # Training dataset load offset in [0, 1].
# i.e. 0.1 means the first 10% data are skipped during training.
VALIDATE_LOAD_PORTION = 1.0
VALIDATE_LOAD_OFFSET = VALIDATE_LOAD_PORTION * 0.0
ONLY_BATCH_SHUFFLE = True  # Whether only shuffle within the music segment samples.
# If "False", then all the frames in each sample will also be shuffled.
OVERLAP_SIZE = None  # Training overlap portion in [0, 1]. "None" means no overlapping in training dataloader.
SAMPLE_SEED = 14  # Global random seed
VELOCITY_RESCALE = False  # Scaling back the velocity value from [0, 1] to [0, 127]
PREPROCESS = 'cqt'  # Spectrogram preprocessing type
CACHE_PIECE = 100  # Every number of "CACHE_PIECE" piano music are packed and saved in one cache file.
# Reduce this parameter if you encounter the CPU memory overflowing problem during data preprocessing.
RESAMPLE_FREQ_DOTS = 16000  # Resample rate
FRAME_LENGTH_MS = 128  # Frame length in millisecond
HOP_LENGTH_RATIO = 0.15625  # Frame hop length portion (%) comparing to the complete frame length.
N_FFT = 2048  # FFT length
N_MEL = 229  # Number of the mel bins
TOP_DB = 80  # dB dynamic range form 0 to "-TOP_DB"
PEDAL_TH = 64  # Normal pedal threshold
N_WORKERS = 0  # Workers in torch.Dataloader
N_JOBS = -1  # Workers for joblib.Parallel

# NDM (note duration modification) params. Please refer to the MT-FiLM paper for detailed explanation.
MAX_PEDAL_SPEED = 0
MIN_PEDAL_RANGE = 1
PEDAL_SPEED_RANGE = 2
PEDAL_MAXIMUM_ACCELERATED_SPEED = 0
NOTE_MINIMUM_FRAME = 40

# postprocess params
ONS_TH = 0.5  # Onset threshold
FRM_TH = 0.5
OFF_TH = 0.5
MIN_MIDI = 21

# model directory
MODEL_NAME = 'MT_FiLM'
MODEL_NAME_APPENDIX = ''  # Optional, only for model name identification
MODEL_PATH = "/home_address/model/"
LOG_PATH = MODEL_PATH + "log/"
CKP_PATH = MODEL_PATH + "ckp/"
CHECKPOINT = None
OPT_CKP = CHECKPOINT
NEW_MODEL_NAME = False
NEW_LOG_TIME = False
NEW_DATA_EPOCH = True  # Refresh the batch informs when reloading from a checkpoint.
# If set as "False", then usually for intra epoch continue training.

# data directory
DATASET_NAME = 'maestro-v3.0.0'
DATASET_PATH = "/home_address/dataset/"
RAW_WAV_PATH = DATASET_PATH
MIDI_PATH = DATASET_PATH
JSON_PATH = DATASET_PATH
CACHE_NAME = "cache_h20ms_cqt352_allPRs"
CACHE_PATH = DATASET_PATH + CACHE_NAME + '/'
VALIDATE_CACHE_PATH = CACHE_PATH

# train & test
FINE_TUNING = False  # Whether fine-tuning form a model checkpoint
WITH_TRAIN = True  # Whether include the training process
WITH_VALIDATE = True  # Whether include the validation process
FORMAL_TRAINING = True  # Set "False" to avoid creating an address path when debugging.

# evaluation
EVA_LOG_PATH = "/home_address/eva/"
EVA_FROM_LOCAL = False  # Evaluation from existing output and ground truth piano rolls.
EVA_ONS_TH = []  # thresholds list for threshold grid searching.
EVA_FRM_TH = []
EVA_OFF_TH = []

