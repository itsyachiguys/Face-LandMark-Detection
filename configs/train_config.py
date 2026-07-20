from pathlib import Path

# -----------------------
# Paths
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_XML = DATA_DIR / "labels_ibug_300W_train.xml"
TEST_XML = DATA_DIR / "labels_ibug_300W_test.xml"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

# -----------------------
# Image
# -----------------------

IMAGE_SIZE = 224

# -----------------------
# Training
# -----------------------

BATCH_SIZE = 32

NUM_WORKERS = 0

EPOCHS = 10

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

# -----------------------
# Model
# -----------------------

NUM_LANDMARKS = 68

DEVICE = "cuda"