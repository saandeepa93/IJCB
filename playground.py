import difflib
from icecream import ic
import datetime
import pandas as pd
from sys import exit as e

# HANDLE TED FRAMES
openface_path = "./data/face_view_v3_of.csv"
df_of = pd.read_csv(openface_path)
# valid_ind = df_of.index[df_of['success'] == 1].tolist()
valid_ind = df_of[df_of['success'] == 1]['frame'].tolist()
valid_ind = [ind-1 for ind in valid_ind]

ted_path = "./data/face_view_v3_ted.csv"
df = pd.read_csv(ted_path)
# SELECT VALID OPENFACE RECORD
# df = df[df.index.isin(valid_ind)]
df = df[df['frame'].isin(valid_ind)]
ind = df['frame'].tolist()

sorted_df = df.sort_values(['ted'], ascending=False)
sorted_df = sorted_df.iloc[::2]
top_40 = sorted_df.iloc[:self.affect_frames]['frame'].tolist()
bottom_10 = sorted_df.iloc[-self.non_affect_frames:]['frame'].tolist()
