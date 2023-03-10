import os
from tqdm import tqdm 
import copy
import json

from icecream import ic
from sys import exit as e


if __name__ == "__main__":
  affect_dir = "/data/dataset/ca_data/affect_split_videos"
  annot_dir = "/data/dataset/ca_data/affect_annot"

  annot_dict = {}

  valence_dict = {"positive": 1, "negative": -1, "same": 0, "null": None}
  expression_dict = {"neutral": 0, "angry": 1, "happy": 2, "sad": 3, "fear": 4, "disgust": 5}

  sublabel_dict = {
                    'v1': {'valence': 0, 'arousal': 0, 'expression': []},
                    'v2': {'valence': 0, 'arousal': 0, 'expression': []},
                    'v3': {'valence': 0, 'arousal': 0, 'expression': []},
                    'v4': {'valence': 0, 'arousal': 0, 'expression': []}
                  }

  pbar = tqdm(os.listdir(annot_dir))
  for sub in pbar:
    sub_dir = os.path.join(annot_dir, sub)
    if not os.path.isdir(sub_dir):
      continue
    
    annot_dict[sub] = {}
    for sess in os.listdir(sub_dir):
      sess_dir = os.path.join(sub_dir, sess)
      if not os.path.isdir(sess_dir):
        continue
      sess_num = sess.split("_")[0]
      annot_dict[sub][sess_num] = copy.deepcopy(sublabel_dict)

      for annot in os.listdir(sess_dir):
        annot_file = os.path.join(sess_dir, annot)
        if os.path.splitext(annot_file)[-1] != ".txt":
          continue
        
        labels = []
        text_file =  open(annot_file, 'r')
        labels = text_file.read().splitlines()
        labels = list(filter(None, labels))
        
        valence = valence_dict[labels[-1]]
        arousal = labels[-2]
        expression = labels[:-2]

        vnum = annot.split('_')[-1].split('.')[0]
        annot_dict[sub][sess_num][vnum]['valence'] = valence
        annot_dict[sub][sess_num][vnum]['arousal'] = arousal
        annot_dict[sub][sess_num][vnum]['expression'] = expression
  
  with open(f'./data/affect_label.json', 'w') as fp:
    json.dump(annot_dict, fp, indent=4)
