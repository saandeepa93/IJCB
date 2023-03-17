import torch 
from torch import nn 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from decord import VideoReader, AudioReader
from decord import cpu, gpu

import albumentations as A
from  albumentations.pytorch.transforms import ToTensorV2

import os
import numpy as np
import pandas as pd
import mediapipe as mp
from sys import exit as e
from icecream import ic

from utils import Iterator

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def crop_faces(img):
  face_detection_results = face_detection.process(img)
  h, w, c = img.shape
  ic(face_detection_results.detections, img.shape)
  e()
  if face_detection_results.detections:
    # Iterate over the found faces.
    for face_no, face in enumerate(face_detection_results.detections):
      face_data = face.location_data.relative_bounding_box
      xleft = face_data.xmin*w
      xleft = int(xleft)
      xtop = face_data.ymin*h
      xtop = int(xtop)
      xright = face_data.width*w + xleft
      xright = int(xright)
      xbottom = face_data.height*h + xtop
      xbottom = int(xbottom)
      return [(xleft, xtop, xright, xbottom)]



class VideoLoader(Dataset):
  def __init__(self, cfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.root_dir = cfg.PATHS.VID_DIR
    self.ted_dir = cfg.PATHS.TED_DIR
    self.aligned_dir = os.path.join(cfg.PATHS.OPENFACE_DIR, "aligned")
    self.iterator = Iterator(self.root_dir, ".mp4", cfg.DATASET.CAMERA_VIEW)
    self.all_files = self.iterator.__getallfiles__()
    self.transforms = self.get_augmentation()

  def get_augmentation(self):
    video_dict = {}
    for i in range(1, self.cfg.DATASET.NUM_FRAMES):
      video_dict[f'image{i}'] = "image"

    # if self.mode == "TRAIN":
    transforms = A.Compose([
      A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
      A.Normalize((0, 0, 0), (1, 1, 1), p=1),
      ToTensorV2()
      ], 
      # keypoints_params = A.KeypointParams(format='xy'), 
      additional_targets=video_dict, p=1)
    
    return transforms
    
  def __len__(self):
    return len(self.all_files)
  
  def __getitem__(self, idx):
    vid_path = self.all_files[idx]
    split_lst = vid_path.split('/')
    subject = split_lst[-3]
    session = split_lst[-2]
    fname = split_lst[-1].split('.')[0]

    ted_path = os.path.join(self.ted_dir, subject, session, f"{fname}.csv")
    aligned_path = os.path.join(self.aligned_dir, subject, session, fname)

    df = pd.read_csv(ted_path)
    sorted_df = df.sort_values(['ted'], ascending=False)
    sorted_df = sorted_df.iloc[::2]
    top_40 = sorted_df.iloc[:40]['frame'].tolist()
    bottom_10 = sorted_df.iloc[-10:]['frame'].tolist()
    
    frames = sorted(top_40 + bottom_10)

    # Video Augmentation
    augment_dict = {}
    for f, frame in enumerate(frames, 0):
      img_path = os.path.join(aligned_path, f"frame_det_00_{str(frame).zfill(6)}.bmp")
      im = np.array(Image.open(img_path))
      crop_im = im.copy()
      if f == 0:
        augment_dict["image"] = crop_im
      else:
        augment_dict[f"image{f}"] = crop_im

    transformed_data = self.transforms(**augment_dict)
    image_list = []
    for f in range(0, self.cfg.DATASET.NUM_FRAMES):
      if f == 0:
        im = transformed_data['image']
      else:
        im = transformed_data[f"image{f}"]
      image_list.append(im)
    x_img = torch.stack(image_list)
    
    # vr = VideoReader(vid_path, ctx=cpu(0))
    # vid = vr.get_batch(frames).asnumpy()
    # for c, frame in enumerate(vid, 0):
    #   detected_faces = crop_faces(frame)
    #   cropped_img = Image.fromarray(frame).crop(detected_faces[0])
    #   ic(cropped_img.shape)

    return x_img





  


