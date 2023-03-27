import difflib
from icecream import ic
import datetime
import pandas as pd
from sys import exit as e

from utils import calculate_ted_score

def get_target_mat(df, target_AUs):
  """
  Return processed OpenFace file 
  """
  col_names = df.columns.tolist()
  
  # create a new dataframe 
  # TODO - take care of the naming issue 
  meta = df.iloc[:, :col_names.index('success')+1]
  gaze_loc = df.iloc[:, col_names.index('gaze_0_x'):col_names.index('gaze_1_z')+1]
  gaze_rot = df.iloc[:, col_names.index('gaze_angle_x'):col_names.index('gaze_angle_y')+1]
  pose_loc = df.iloc[:, col_names.index('pose_Tx'):col_names.index('pose_Tz')+1]
  pose_rot = df.iloc[:, col_names.index('pose_Rx'):col_names.index('pose_Rz')+1]
  landmarks = df.iloc[:, col_names.index('x_0'):col_names.index('y_67')+1]
  au_inten_OFPAU = df[target_AUs]

  return meta, gaze_loc, gaze_rot, pose_loc, pose_rot, landmarks, au_inten_OFPAU

au_int = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
au_int_cols = [f"AU{str(AU).zfill(2)}_r" for AU in au_int]
of_df = pd.read_csv("./data/face_view_v3.csv", encoding = "ISO-8859-1")
meta, gaze_loc, gaze_rot, pose_loc, pose_rot, landmarks, au_inten_OFPAU = get_target_mat(of_df, au_int_cols) 
ted_score = calculate_ted_score(au_inten_OFPAU.values, gaze_loc.values, \
                                    gaze_rot.values, pose_loc.values, pose_rot.values,\
                                        landmarks.values, 10)
                                  
ted_score.columns = ['ted']
ted_score['frame'] = range(0, ted_score.shape[0])

ted_score.to_csv("./face_view_v3.csv")
# plot_ted(ted_score, os.path.join(dest_sess_dir, f"{fname.split('.')[0]}.html"))

e()

# HANDLE TED FRAMES
openface_path = "./data/face_view_v3.csv"
df_of = pd.read_csv(openface_path)
ic(df_of.head())
e()
# valid_ind = df_of.index[df_of['success'] == 1].tolist()
valid_ind = df_of[df_of['success'] == 1]['frame'].tolist()
ic(df_of['success'])
e()
valid_ind = [ind-1 for ind in valid_ind]
ic(len(valid_ind))
e()

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
