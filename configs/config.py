from yacs.config import CfgNode as CN


_C = CN()

# PATHS
_C.PATHS = CN()
_C.PATHS.VID_DIR = ""
_C.PATHS.OPENFACE_DIR = ""
_C.PATHS.TED_DIR = ""
_C.PATHS.VIS_PATH = ""

# TED
_C.TED = CN()
_C.TED.WINDOW = 10

# JSON FILES
_C.JSON = CN()
_C.JSON.TOUCH_TS = ""

# DATASET
_C.DATASET = CN()
_C.DATASET.N_CLASS = 2
_C.DATASET.IMG_SIZE = 224
_C.DATASET.NUM_WORKERS = 4
_C.DATASET.CAMERA_VIEW = "face"
_C.DATASET.NUM_FRAMES = 50

# SPLIT
_C.SPLIT = CN()
_C.SPLIT.SUBJECT = "P0009"
_C.SPLIT.VAL_SPLIT = "sess"
_C.SPLIT.SPLIT_VALUE = "S3"

# TRAINING
_C.TRAINING = CN()
_C.TRAINING.BATCH = 32

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()