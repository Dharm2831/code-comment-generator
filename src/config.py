"""All project settings in one place. Change values here, not inside the other files."""

# ---- Model and dataset -------------------------------------------------------
BASE_MODEL = "Salesforce/codet5-small"            # pretrained Transformer (about 60M parameters)
DATASET_NAME = "google/code_x_glue_ct_code_to_text"  # CodeXGLUE code-to-text (from CodeSearchNet)
LANGUAGES = ["python", "java"]

# ---- How much data to use (per language) -------------------------------------
# The full dataset has ~250k Python and ~165k Java functions. A subset keeps
# training short on a laptop. Raise TRAIN_PER_LANG for better quality.
TRAIN_PER_LANG = 30_000
VAL_PER_LANG = 1_000
TEST_PER_LANG = 1_000

# ---- Input / output sizes (in tokens) ----------------------------------------
MAX_SOURCE_LEN = 256   # longer functions are cut off here
MAX_TARGET_LEN = 48    # comments are short

# ---- Training hyperparameters ------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1     # first 10% of steps: learning rate rises from 0
GRAD_CLIP = 1.0
SEED = 42

# ---- Generation --------------------------------------------------------------
NUM_BEAMS = 5

# ---- Folders -----------------------------------------------------------------
DATA_DIR = "dataset"                       # dataset is downloaded here
MODEL_DIR = "models/comment-generator"     # trained model is saved here
RESULTS_DIR = "results"                    # graphs and numbers for the report
