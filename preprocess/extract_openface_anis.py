
import subprocess
import os
import shutil
from tqdm import tqdm
from pathlib import Path
from sys import exit as e
from icecream import ic 
from tqdm import tqdm
import glob
import pickle


FNULL = open(os.devnull, 'w')
def sh(cmd):
  return subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)


def ex_openface(openface_path, context_path):
  args = (openface_path, "-f", context_path)
  stat = sh(args)
  return stat

def make_directories(path):
  if not os.path.isdir(path):
    os.mkdir(path)


if __name__ == "__main__":
  # Input and binaries
  root_dir = "/home/saandeepaath/Downloads/relabeled_affectnet/relabeled_affectnet/valid"

  # Destination
  dest_root = "/data/dataset/anis_temp/"
  
  dest_csv_root = os.path.join(dest_root, "csv")
  dest_aligned_root = os.path.join(dest_root, "aligned")
  dest_vid_root = os.path.join(dest_root, "viz")

  # Openface
  openface_dir = "/home/saandeepaath/Desktop/OpenFace/build"
  binaries = "bin/FaceLandmarkImg"
  openface_path = os.path.join(openface_dir, binaries)
  output_path = "./processed"

  failed = []

  pbar = tqdm(os.listdir(root_dir))
  for subject in pbar:
    subject_dir = os.path.join(root_dir, subject)

    # Define root destination directories
    dest_csv_subject = os.path.join(dest_csv_root, subject)
    dest_aligned_subject = os.path.join(dest_aligned_root, subject)
    dest_vid_subject = os.path.join(dest_vid_root, subject)

    # Create directories
    make_directories(dest_csv_subject)
    make_directories(dest_aligned_subject)
    make_directories(dest_vid_subject)

    for img in os.listdir(subject_dir):
      img_path = os.path.join(subject_dir, img)
      img_name = img.split('.')[0]
      if os.path.splitext(img_path)[-1] not in [".png", ".jpeg", ".jpg"]:
        continue

      # Defin source paths
      src_csv = os.path.join(output_path, f"{img_name}.csv")
      src_aligned = os.path.join(output_path, f"{img_name}_aligned/face_det_000000.bmp")
      src_vid = os.path.join(output_path, f"{img_name}.jpg")

      # Define destination paths
      dest_csv = os.path.join(dest_csv_subject, f"{img_name}.csv")
      dest_aligned = os.path.join(dest_aligned_subject, f"{img_name}.png")
      dest_vid = os.path.join(dest_vid_subject, f"{img_name}.jpg")

      pbar.set_description(subject_dir)
        # print("Extracting openface features")
      status = ex_openface(openface_path, img_path)
        
      if not os.path.isfile(src_csv):
        print(f"{img_path} CSV failed to extract openface features")
        failed.append(img_path)
        continue
      if not os.path.isfile(src_aligned):
        print(f"{img_path} ALIGNED failed to extract openface features")
        failed.append(img_path)
        continue
      if not os.path.isfile(src_vid):
        print(f"{img_path} VIZ failed to extract openface features")
        failed.append(img_path)
        continue

      # Copy data
      shutil.copy(src_csv, dest_csv)
      shutil.copy(src_aligned, dest_aligned)
      shutil.copy(src_vid, dest_vid)
      # print("Removing `processed` directory...")
      shutil.rmtree(output_path)
  
  with open("./failed_of.txt", "wb") as fp:   #Pickling
    pickle.dump(failed, fp)




  