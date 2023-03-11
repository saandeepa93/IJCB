import os 
from sys import exit as e
from icecream import ic

import pandas as pd
import json
import yaml




if __name__ == "__main__":
  vars = read_yaml()

  touch_ts_path = vars['json']['touch_ts']
  
  with open(touch_ts_path, 'r') as fp:
    touch_ts_all = json.load(fp)

  valid_subjects = [key for key in touch_ts_all.keys() if touch_ts_all[key]]

