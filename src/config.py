import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_FULL = os.path.join(ROOT_DIR, "data", "full")
DATA_DIR_SAMPLE = os.path.join(ROOT_DIR, "data", "sample")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CLASSES = [
    "MT_Blowhole",
    "MT_Break",
    "MT_Crack",
    "MT_Fray",
    "MT_Uneven",
    "MT_Free",
]

CLASS_DISPLAY_NAMES = {
    "MT_Blowhole": "Blowhole",
    "MT_Break": "Break",
    "MT_Crack": "Crack",
    "MT_Fray": "Fray",
    "MT_Uneven": "Uneven",
    "MT_Free": "No Defect",
}

IMG_SIZE = (128, 128)
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0]

CANNY_LOW = 50
CANNY_HIGH = 150

RANDOM_STATE = 42
TEST_SIZE = 0.2
