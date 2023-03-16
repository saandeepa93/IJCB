from yacs.config import CfgNode as CN


_C = CN()

# PATHS
_C.PATHS = CN()
_C.PATHS.VID_DIR = ""
_C.PATHS.OPENFACE_DIR = ""
_C.PATHS.TED_DIR = ""

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
_C.DATASET.TEST = "P0002"
_C.DATASET.CAMERA_VIEW = "face"

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()