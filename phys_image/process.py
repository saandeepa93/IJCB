import os
from datetime import datetime

import re
import regex

import textract
import pandas as pd

from sys import exit as e
from icecream import ic


shift_dict = {'1': '!', '2': '@', '3': '#', '4': '$', '5': '%', '6': '^', \
              '7': '&', '8': '*', '9': '(', '0': ')', '-': '_', '=': '+'}



#*******************************************GLOBAL helpers*******************************************
def apply_keystroke(filtered_df, filtered_ind, new_df):
  window = filtered_df.rolling(2)
  # window = new_df['keys'].rolling(2, closed = "both")
  ischar_dict = {}
  for ind in filtered_ind:
    ischar_dict[ind] = "NA"

  for t in window:
    if t.shape[0] == 1:
      continue

    key1 = t['keys'].iloc[0].lower()
    key2 = t['keys'].iloc[1].lower()

    if key1 in list(shift_dict.values()):
      key1_isupper = True
    elif key2 in list(shift_dict.values()):
      key2_isupper = True
    else:
      key1_isupper = t['keys'].iloc[0].isupper()
      key2_isupper = t['keys'].iloc[1].isupper()
    
    if key1 == key2:
      if ischar_dict[t.index[0]] == "NA":
        ischar_dict[t.index[0]] = "PRESSED"
        ischar_dict[t.index[1]] = "RELEASED"
    
    elif key1 in shift_dict:
      shift_key1 = shift_dict[key1]
      if shift_key1 == key2: 
        if ischar_dict[t.index[0]] == "NA":
          ischar_dict[t.index[0]] = "PRESSED"
          ischar_dict[t.index[1]] = "RELEASED"
    
    elif key2 in shift_dict:
      shift_key2 = shift_dict[key2]
      if shift_key2 == key1: 
        if ischar_dict[t.index[0]] == "NA":
          ischar_dict[t.index[0]] = "PRESSED"
          ischar_dict[t.index[1]] = "RELEASED"
    
    else:
      ischar_dict[t.index[-1]] = 'NA'

  new_df['type'].iloc[filtered_ind] = list(ischar_dict.values())
  return new_df

def match_pattern(mtc_str, key_str, first=True):
  # Pattern matching
  first_match = None
  pattern = regex.compile(mtc_str)
  matches = pattern.finditer(key_str)
  for match in matches:
    first_match = match.span()
    if first:
      break
    else:
      continue
  return first_match

def prepare_csv_phone(phone_key_file):
  keylog_cols = ['keys', 'type', 'ts']
  df = pd.read_csv(phone_key_file, header=None, names=keylog_cols)
  df = df.dropna()

  df['keys'] = df['keys'].astype(int)
  df = df.drop(df[df['keys'] < 0].index)
  df['chr'] = df['keys'].apply(chr)
  
  df['ts'] = pd.to_datetime(df['ts'], format = '%Y-%m-%d %H:%M:%S.%f')
  # df['ts'] = df['ts'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S,%f'))
  return df

def prepare_csv(desktop_key_file):
  df = pd.read_csv(desktop_key_file, sep='\\n', header=None, names = ['all'], engine='python')
  df = df.dropna()
  # SPLIT DATAFRAME INTO TIMESTAMPS AND KEYS
  new_df = pd.DataFrame(columns = ['ts', 'keys'])
  t = df['all'].str.split('-', -1, expand=True).reset_index()
  new_df['ts'] = t.iloc[:, 1].astype(str) + "-" + t.iloc[:, 2].astype(str) + "-" + t.iloc[:, 3].astype(str)
  new_df['keys'] = t.iloc[:, 4]
  # CONVERT KEYS TO STRING TYPE
  cols = new_df.select_dtypes(['object']).columns
  new_df[cols] = new_df[cols].apply(lambda x: x.str.strip())
  new_df['keys'] = new_df['keys'].str.replace('\'', '')
  # ASSIGN PRESSED OR RELEASED TO CHARS ONLY. IGNORE SPL KEYS FOR NOW
  new_df['ischar'] = new_df.apply(lambda x: "Yes" if len(x['keys']) == 1 \
    else "space" if x['keys'].lower() == "key.space" else "No", axis=1)
  new_df['type'] = "NA"
  # FILL IN THE DETAILS
  uniq_keys = new_df['keys'].unique()
  for k in uniq_keys:
    if len(k) == 1: 
      filtered_ind = new_df.index[new_df['keys']== k].tolist()
      filtered_df = new_df.iloc[filtered_ind]
      filtered_df = filtered_df.reindex(filtered_ind)
      new_df = apply_keystroke(filtered_df, filtered_ind, new_df)
  # new_df.to_csv("./shed/test.csv")
  return new_df


#*******************************************DESKTOP helpers*******************************************
def process_ff_file(desktop_ff_file, desktop_key_file):
  # READ FREE FORM PWD
  text = textract.process(desktop_ff_file).decode()
  
  # RETRIEVE THE FIRST AND LAST VALUE ENTERED
  mtc_str1 = r'personal user (.*?)pass'
  mtc_str2 = r'pass'

  span1 = match_pattern(mtc_str1.lower(), text.lower(), True)
  span2 = match_pattern(mtc_str2.lower(), text.lower(), False)
  
  pass1 = text[span1[0]+13:span1[1]-5].strip()
  pass2 = text[span2[0]+4:].strip()

  
  new_df = prepare_csv(desktop_key_file)
  char_df = new_df[new_df['type'] == "PRESSED"]
  key_lst = char_df['keys'].tolist()
  key_str = ''.join(key_lst)


  e1 = e2 = 1
  c1 = 4
  c2 = 4
  match_found = 0
  while (e1 <= c1 or e2 <= c2) and match_found == 0:
    mtc_str1 = rf"(?:{re.escape(pass1)}){{e<={e1}}}"
    mtc_str2 = rf'(?:{re.escape(pass2)}){{e<={e2}}}'
    
    span1 = match_pattern(mtc_str1.lower(), key_str.lower(), True)
    span2 = match_pattern(mtc_str2.lower(), key_str.lower(), False)
    
    if span1 is None and span2 is None:
      break
    elif span1 is None:
      e1 = e1 + 1
    elif span2 is None:
      e2 = e2 + 1
    else:
      match_found = 1

  if not match_found:
    ic(f"No match found for DESKTOP: FREE-FORM FILE")
    return None, None

  
  try:
    task_begin = new_df[(new_df['ischar'] == 'Yes') & (new_df['type'] == 'PRESSED')]\
      .iloc[span1[0]]['ts']
    task_end = new_df[(new_df['ischar'] == 'Yes') & (new_df['type'] == 'PRESSED')]\
      .iloc[span2[-1]-1]['ts']
    return str(task_begin), str(task_end)
  except:
    ic(desktop_key_file)
    return None, None

def process_fixed_file(desktop_key_file, pwd):
  new_df = prepare_csv(desktop_key_file)
  # MATCH WITH GIVEN TEXT
  # GET SENTENCES
  char_df = new_df[new_df['type'] == "PRESSED"]
  key_lst = char_df['keys'].tolist()
  key_str = ''.join(key_lst)
  
  # MATCH STRING  
  pwd1 = pwd[0]
  pwd2 = pwd[-1]
  mtc_str1 = rf"(?:{re.escape(pwd1)}){{e<=3}}"
  mtc_str2 = rf'(?:{re.escape(pwd2)}){{e<=3}}'

  # mtc_str1 = r'(?:GmxPV3L){e<=3}'
  # mtc_str2 = r'(?:jxK&5sDpwfE\+U){e<=3}'

  span1 = match_pattern(mtc_str1.lower(), key_str.lower(), True)
  span2 = match_pattern(mtc_str2.lower(), key_str.lower(), False)
  
  try:
    task_begin = new_df[(new_df['ischar'] == 'Yes') & (new_df['type'] == 'PRESSED')]\
      .iloc[span1[0]]['ts']
    task_end = new_df[(new_df['ischar'] == 'Yes') & (new_df['type'] == 'PRESSED')]\
      .iloc[span2[-1]-1]['ts']
    return str(task_begin), str(task_end)
  except:
    ic(desktop_key_file)
    return None, None

def process_essay_file(desktop_essay_file, desktop_key_file):
  wrd_strt_count = 4
  wrd_end_count = 4
  # READ KEYLOGGER FILE
  new_df = prepare_csv(desktop_key_file)
  char_df = new_df[new_df['type'] == "PRESSED"]
  key_lst = char_df['keys'].tolist()
  key_str = ''.join(key_lst)

  # READ FREE FORM PWD
  text = textract.process(desktop_essay_file).decode()
  search_text_strt = ''.join(text.split(' ')[:wrd_strt_count])
  search_text_end = ''.join(text.split(' ')[-wrd_end_count:])

  e1 = e2 = 1
  c1 = c2 = 6

  match_found = 0
  while (e1 <= c1 or e2 <= c2) and match_found == 0:
    mtc_str1 = rf"(?:{re.escape(search_text_strt)}){{e<={e1}}}"
    mtc_str2 = rf'(?:{re.escape(search_text_end)}){{e<={e2}}}'
    
    span1 = match_pattern(mtc_str1.lower(), key_str.lower(), True)
    span2 = match_pattern(mtc_str2.lower(), key_str.lower(), False)
    
    if span1 is None and span2 is None:
      e1 += 1
      e2 += 1
    elif span1 is None:
      e1 = e1 + 1
    elif span2 is None:
      e2 = e2 + 1
    else:
      match_found = 1

  if not match_found:
    ic(f"No match found for DESKTOP: ESSAY FILE")
    return None, None


  try:
    task_begin = new_df[(new_df['ischar'] == 'Yes') & (new_df['type'] == 'PRESSED')]\
      .iloc[span1[0]]['ts']
    task_end = new_df[(new_df['ischar'] == 'Yes') & (new_df['type'] == 'PRESSED')]\
      .iloc[span2[-1]-1]['ts']
    return str(task_begin), str(task_end)
  except:
    ic(desktop_key_file)
    return None, None


#*******************************************SMARTPHONE helpers*******************************************
def process_fixed_file_phone(phone_key_file, pwd, session_start, mod_flg=0):
  new_df = prepare_csv_phone(phone_key_file)
  # APPLIES ONLY TO P0011 S3
  if mod_flg:
    new_df['ts'] = new_df['ts'] + pd.Timedelta(seconds=3000)
  session_start = datetime.strptime(session_start, '%Y-%m-%d %H:%M:%S,%f')
  new_df = new_df[new_df['ts'] >= session_start]
  
  # MATCH WITH GIVEN TEXT
  char_df = new_df[new_df['type'] == "PRESSED"]
  key_lst = char_df['chr'].tolist()
  key_str = ''.join(key_lst)
  
  pwd1 = pwd[0]
  pwd2 = pwd[-1]

  e1 = e2 = 1
  c1 = c2 = 6

  match_found = 0
  while (e1 <= c1 or e2 <= c2) and match_found == 0:
    mtc_str1 = rf"(?:{re.escape(pwd1)}){{e<={e1}}}"
    mtc_str2 = rf'(?:{re.escape(pwd2)}){{e<={e2}}}'
    
    span1 = match_pattern(mtc_str1.lower(), key_str.lower(), True)
    span2 = match_pattern(mtc_str2.lower(), key_str.lower(), False)
    
    if span1 is None and span2 is None:
      e1 += 1
      e2 += 1
    elif span1 is None:
      e1 = e1 + 1
    elif span2 is None:
      e2 = e2 + 1
    else:
      match_found = 1

  if not match_found:
    ic(f"No match found for PHONE: FIXED FILE")
    return None, None

  try:
    task_begin = new_df[(new_df['type'] == 'PRESSED')]\
      .iloc[span1[0]]['ts']
    task_end = new_df[(new_df['type'] == 'PRESSED')]\
      .iloc[span2[-1]-1]['ts']
    
    return str(task_begin), str(task_end)
  except:
    ic(phone_key_file, session_start)
    return None, None

def process_ff_file_phone(phone_ff_file, phone_key_file, session_start, mod_flg=0):
  new_df = prepare_csv_phone(phone_key_file)
  # APPLIES ONLY TO P0011 S3
  if mod_flg:
    new_df['ts'] = new_df['ts'] + pd.Timedelta(seconds=mod_flg)
  session_start = datetime.strptime(session_start, '%Y-%m-%d %H:%M:%S,%f')
  new_df = new_df[new_df['ts'] >= session_start]
  
  # MATCH WITH GIVEN TEXT
  char_df = new_df[new_df['type'] == "PRESSED"]
  key_lst = char_df['chr'].tolist()
  key_str = ''.join(key_lst)


  # READ FREE FORM PWD
  text = textract.process(phone_ff_file).decode()
  mtc_str1 = r'personal_user\n(.*?)\npersonal_pass'
  mtc_str2 = r'school_pass'

  span1 = match_pattern(mtc_str1.lower(), text.lower(), True)
  span2 = match_pattern(mtc_str2.lower(), text.lower(), False)

  
  # GET FREE FORM IN CASE OUTPUT FORMAT DOES NOT HAVE NAMES
  lines = text.split('\n')
  if len(lines) <=11:
    # pass1 = lines[0]
    # pass2 = lines[-1]
    pass1 = next(s for s in lines if s)
    pass2 = next(s for s in reversed(lines) if s)
  else:
    pass1 = text[span1[0]+13:span1[1]-13].strip()
    pass2 = text[span2[0]+11:].strip()

  e1 = e2 = 1
  c1 = c2 = 6
  match_found = 0
  while (e1 <= c1 or e2 <= c2) and match_found == 0:
    mtc_str1 = rf"(?:{re.escape(pass1)}){{e<={e1}}}"
    mtc_str2 = rf'(?:{re.escape(pass2)}){{e<={e2}}}'
    
    span1 = match_pattern(mtc_str1.lower(), key_str.lower(), True)
    span2 = match_pattern(mtc_str2.lower(), key_str.lower(), False)
    
    if span1 is None and span2 is None:
      e1 += 1
      e2 += 1
    elif span1 is None:
      e1 = e1 + 1
    elif span2 is None:
      e2 = e2 + 1
    else:
      match_found = 1

  if not match_found:
    ic(f"No match found for DESKTOP: FREE-FORM FILE")
    return None, None

  
  try:
    task_begin = new_df[(new_df['type'] == 'PRESSED')]\
      .iloc[span1[0]]['ts']
    task_end = new_df[(new_df['type'] == 'PRESSED')]\
      .iloc[span2[-1]-1]['ts']
    return str(task_begin), str(task_end)
  except:
    ic(phone_ff_file)
    return None, None


#*******************************************PHYSIOLOGY helpers*******************************************
def process_affect_files():
  pass

# ic(process_fixed_file("./key_logs.txt"))
# ic(process_ff_file("./task_free_form_pass.docx", "./key_logs.txt"))
# ic(process_essay_file("./task_essay.docx", "./key_logs.txt"))
# ic(process_fixed_file_phone("./keylogger.txt", ["schoolRocks", "GmxPV3L"], "2022-11-14 09:54:46,133"))
# ic(process_fixed_file_phone("./keylogger.txt", ["GmxPV3L", "jxK&5sDpwfE+U"], "2022-05-26 09:55:02,313", 0))
# ic(process_ff_file_phone("ca_home_act_2.txt", "./keylogger.txt", "2022-05-26 09:55:02,313", 0))

  