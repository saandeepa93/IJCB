# **Face and Physiology based Continuous Authentication System**

## **Setup**
```
pip install -r requirements.txt
```

## **Train Images only**
```
CUDA_VISIBLE_DEVICES=0 python ./trainer/train_images.py --config <config_name>
```
  * Write your own `yaml` config file and place it in the path `./configs/experiments`

## **Directory Structure**

  * `configs`: All the config files are present here under the `experiments` directory
  * `loader`: All the dataloader for csv, video and physiology signals
  * `models`: Models pertaining to classification
  * `phys_image`: Contains physiology and image based preprocessing. Refer to README inside the directory for more details
  * `preprocess`: Dataset preparation scripts. Execute in order mentioned
  * `trainer`: All the training scripts are present here
  * `utils`: Helpers scripts utilized throughout the code

