import os 
import numpy as np
import datetime
import pandas as pd
import copy
import json
import yaml

import unicodedata
from tqdm import tqdm

from icecream import ic
from sys import exit as e

import process as p


def read_yaml():
  with open("./variables.yaml", "r") as stream:
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == "__main__":
  # READ VARIABLES
  vars = read_yaml()

  # ROOT DIRECTORIES
  touch_dir = vars['paths']['touch']
  affect_annot_dir = vars['paths']['affect_annot']
  affect_ts_path = vars['json']['affect_ts']

  # CHILDREN
  children = ['P0030', 'P0031', 'P0039']  
  error = {'P0029': ['S1', 'S2', 'S3'], 'P0016': ['S3']}

  # TOUCH VARIABLES
  desktop_key = "keylogger_desktop"
  phone_key = "keylogger_phone"
  desktop_mouse = "mouselogger_desktop"
  desktop_ca = "CA_output_desktop"
  phone_ca = "CA_output_phone"

  # AFFECT DURATIONS
  sess_duration = [
    [84, 60, 58, 19],
    [67, 61, 66, 49],
    [59, 58, 59, 57]
  ]


  # PER SESSION INFO
  task_timestamps = {
                      "desktop": 
                        {
                          "free_form": {"start":0, "end":0}, 
                          "essay": {"start":0, "end":0},
                          "fixed":{"start":0, "end":0}
                        },
                      "phone": 
                        {
                          "fixed": {"start":0, "end":0},
                          "free_form": {"start":0, "end":0}
                        }, 
                      "affect": 
                        {
                          "vid1": {"start":0, "end":0},
                          "vid2": {"start":0, "end":0},
                          "vid3": {"start":0, "end":0},
                          "vid4": {"start":0, "end":0}
                        },
                      "session_start": 0
                }

  # STORE ALL INFO IN JSON
  subject_task_timestamps = {}

  # GET AFFECT TIMESTAMPS
  with open(affect_ts_path, 'r') as fp:
    affect_ts_all = json.load(fp)
      

  pbar = tqdm(os.listdir(touch_dir))
  for sub in pbar:
    touch_sub_dir = os.path.join(touch_dir, sub)
    affect_annot_sub_dir = os.path.join(affect_annot_dir, sub)
    if not os.path.isdir(touch_sub_dir):
      continue
    
    subject_task_timestamps[sub] = {}
    
    child_flag = 0
    if sub in children:
      child_flag = 1

    # FOR EACH SESSION
    for sess in os.listdir(touch_sub_dir):
      touch_sess_dir = os.path.join(touch_sub_dir, sess)
      affect_annot_sess_dir = os.path.join(affect_annot_sub_dir, sess)
      if not os.path.isdir(touch_sess_dir):
        continue  

      sess_name = sess.split('_')[0]
      if sub in error:
        if sess_name in error[sub]:
          ic(sub, sess_name)
          continue
      
      temp = copy.deepcopy(task_timestamps)
      subject_task_timestamps[sub][sess] = temp
      
      desktop_key_dir = os.path.join(touch_sess_dir, desktop_key)
      phone_key_dir = os.path.join(touch_sess_dir, phone_key)
      desktop_ca_dir = os.path.join(touch_sess_dir, desktop_ca)
      phone_ca_dir = os.path.join(touch_sess_dir, phone_ca)


      #*************************************************DESKTOP*************************************************************
      # TASK 4: DESKTOP FIXED PWD
      desktop_key_file = os.path.join(desktop_key_dir, "key_logs.txt")
      if not os.path.isfile(desktop_key_file):
        print(f"{desktop_key_file} does not exist")
        continue
      pbar.set_description(desktop_key_file)
      if child_flag:
        pwd = list(vars['fixed_pwd']['children'].values())
      else:
        pwd = list(vars['fixed_pwd']['adult'].values())
      ts_start, ts_end = p.process_fixed_file(desktop_key_file, pwd)
      subject_task_timestamps[sub][sess]["desktop"]["fixed"]["start"] = ts_start
      subject_task_timestamps[sub][sess]["desktop"]["fixed"]["end"] = ts_end
      
      # TASK 1: DESKTOP FREE FORM PWD
      desktop_ff_file = os.path.join(desktop_ca_dir, "task_free_form_pass.docx")
      if not os.path.isfile(desktop_ff_file):
        print(f"{desktop_ff_file} does not exist")
        subject_task_timestamps[sub][sess]["desktop"]["free_form"]["start"] = "NA"
        subject_task_timestamps[sub][sess]["desktop"]["free_form"]["end"] = "NA"
      else:
        ts_start2, ts_end2 = p.process_ff_file(desktop_ff_file, desktop_key_file)
        subject_task_timestamps[sub][sess]["desktop"]["free_form"]["start"] = ts_start2
        subject_task_timestamps[sub][sess]["desktop"]["free_form"]["end"] = ts_end2
        subject_task_timestamps[sub][sess]["session_start"] = ts_start2
      

      # TASK 2: DESKTOP ESSAY
      desktop_essay_file = os.path.join(desktop_ca_dir, "task_essay.docx")
      if not os.path.isfile(desktop_essay_file):
        print(f"{desktop_essay_file} does not exist")
        subject_task_timestamps[sub][sess]["desktop"]["essay"]["start"] = "NA"
        subject_task_timestamps[sub][sess]["desktop"]["essay"]["end"] = "NA"
      else:
        ts_start3, ts_end3 = p.process_essay_file(desktop_essay_file, desktop_key_file)
        subject_task_timestamps[sub][sess]["desktop"]["essay"]["start"] = ts_start3
        subject_task_timestamps[sub][sess]["desktop"]["essay"]["end"] = ts_end3
      
      
      #*************************************************PHONE*************************************************************
      phone_key_file = os.path.join(phone_key_dir, "keylogger.txt")
      # TASK 1: SMARTPHONE FIXED
      if not os.path.isfile(phone_key_file):
        print(f"{phone_key_file} does not exist")
        subject_task_timestamps[sub][sess]["phone"]["fixed"]["start"] = "NA"
        subject_task_timestamps[sub][sess]["phone"]["fixed"]["end"] = "NA"
      if ts_start2 is None:
        subject_task_timestamps[sub][sess]["phone"]["fixed"]["start"] = "NA"
        subject_task_timestamps[sub][sess]["phone"]["fixed"]["end"] = "NA"
      else:
        ts_start4, ts_end4 = p.process_fixed_file_phone(phone_key_file, pwd, ts_start2)
        subject_task_timestamps[sub][sess]["phone"]["fixed"]["start"] = ts_start4
        subject_task_timestamps[sub][sess]["phone"]["fixed"]["end"] = ts_end4

      # TASK 2: SMARTPHONE FREE-FORM
      phone_ff_file = os.path.join(phone_ca_dir, "ca_home_act_2.txt")
      if not os.path.isfile(phone_ff_file):
        print(f"{phone_ff_file} does not exist")
        subject_task_timestamps[sub][sess]["phone"]["free_form"]["start"] = "NA"
        subject_task_timestamps[sub][sess]["phone"]["free_form"]["end"] = "NA"
      if ts_start2 is None:
        subject_task_timestamps[sub][sess]["phone"]["free_form"]["start"] = "NA"
        subject_task_timestamps[sub][sess]["phone"]["free_form"]["end"] = "NA"
      else:
        ts_start5, ts_end5 = p.process_ff_file_phone(phone_ff_file, phone_key_file, ts_start2)
        subject_task_timestamps[sub][sess]["phone"]["free_form"]["start"] = ts_start5
        subject_task_timestamps[sub][sess]["phone"]["free_form"]["end"] = ts_end5
      
  
      #*************************************************AFFECT*************************************************************
      if ts_end5 is None:
        continue
      convert_to_datetime = lambda x: datetime.datetime.strptime(x, '%H:%M:%S')
      approx_affect_start_time = datetime.datetime.strptime(ts_end5, '%Y-%m-%d %H:%M:%S.%f')


      # GET SESSION VIDEO DURATION
      sess_num = int(sess.split('_')[0][-1]) - 1
      duration = sess_duration[sess_num]

      # RETREIVE AFFECT TS FOR SPECIFIC SUB AND SESS
      try:
        affect_ts = affect_ts_all[sub][sess.split('_')[0]]['tv_view']
      except:
        continue

      # VIDEO 1
      v1_start_time = approx_affect_start_time + datetime.timedelta(0, 45)
      v1_end_time = v1_start_time + datetime.timedelta(0, duration[0])
      subject_task_timestamps[sub][sess]["affect"]["vid1"]["start"] = str(v1_start_time)
      subject_task_timestamps[sub][sess]["affect"]["vid1"]["end"] = str(v1_end_time)

      # VIDEO 2
      a1_end = convert_to_datetime(affect_ts['vid1'][-1])
      a2_start = convert_to_datetime(affect_ts['vid2'][0])
      t1 = (a2_start-a1_end).total_seconds()
      if t1 < 0:
        ic(affect_ts, sub, sess, "VID1")
      v2_start_time = v1_end_time + datetime.timedelta(0, t1)
      v2_end_time = v2_start_time + datetime.timedelta(0, duration[1])
      subject_task_timestamps[sub][sess]["affect"]["vid2"]["start"] = str(v2_start_time)
      subject_task_timestamps[sub][sess]["affect"]["vid2"]["end"] = str(v2_end_time)

      # VIDEO 3
      a2_end = convert_to_datetime(affect_ts['vid2'][-1])
      a3_start = convert_to_datetime(affect_ts['vid3'][0])
      t2 = (a3_start-a2_end).total_seconds()
      if t2 < 0:
        ic(affect_ts, sub, sess, "VID2")
      v3_start_time = v2_end_time + datetime.timedelta(0, t2)
      v3_end_time = v3_start_time + datetime.timedelta(0, duration[2])
      subject_task_timestamps[sub][sess]["affect"]["vid3"]["start"] = str(v3_start_time)
      subject_task_timestamps[sub][sess]["affect"]["vid3"]["end"] = str(v3_end_time)
      
      # VIDEO 4
      a3_end = convert_to_datetime(affect_ts['vid3'][-1])
      a4_start = convert_to_datetime(affect_ts['vid4'][0])
      t3 = (a4_start-a3_end).total_seconds()
      if t3 < 0:
        ic(affect_ts, sub, sess, "VID3")
      v4_start_time = v3_end_time + datetime.timedelta(0, t3)
      v4_end_time = v4_start_time + datetime.timedelta(0, duration[3])
      subject_task_timestamps[sub][sess]["affect"]["vid4"]["start"] = str(v4_start_time)
      subject_task_timestamps[sub][sess]["affect"]["vid4"]["end"] = str(v4_end_time)




  with open('./data/touch_ts_with_affect.json', 'w') as fp:
    json.dump(subject_task_timestamps, fp,  indent=4)
  # ic(subject_task_timestamps)



      