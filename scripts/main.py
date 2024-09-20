import os
import sys
import cv2
import copy
import json
import random
import argparse
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from typing import Tuple, Dict, List

from ultralytics import YOLO
from roboflow import Roboflow

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from sklearn.model_selection import train_test_split

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image, ImageFile, ImageFont, ImageDraw, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import monodepth2.networks as networks
from monodepth2.utils import download_model_if_doesnt_exist

from utils import *
from model import YOLOv9


FUSED_SHAPE = (1280, 640)



if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv9().to(device)
    encoder, depth_decoder, loaded_dict_enc = get_mono_640x192_model()

    # image = np.array(Image.open(test_image_path_list[i])).transpose(2, 0, 1)[:3]

    # disp_resized_np, pred_image = prediction(image, 
    #                                          model,
    #                                          encoder,
    #                                          depth_decoder,
    #                                          loaded_dict_enc)
    
    
    video_path = 'videos/track.mp4'
    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Размер видео: {frame_width}x{frame_height}, FPS: {fps}, Количество кадров: {total_frames}")

    output_video_path = 'output_track.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    num_frames = 0
    frame_count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()

        frame_count += 1
        num_frames += 1
        current_fps = calculate_fps(start_time, num_frames)
        
        if frame_count % 5 == 0 or frame_count == 1:
            if ret == True:
                disp_resized_np, annotated_frame = process_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 
                                                                model,
                                                                encoder,
                                                                depth_decoder,
                                                                loaded_dict_enc)
                cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Frame', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
        else:
            cv2.imshow('Frame', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))    
        
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        # time.sleep(1 / fps / 5)    

    cap.release()
    out.release()

    cv2.destroyAllWindows()
