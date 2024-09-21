import os
import torch
import time
import copy
import numpy as np

from torchvision import transforms

from PIL import Image, ImageDraw
from typing import List

import monodepth2.networks as networks
from monodepth2.utils import download_model_if_doesnt_exist



COLORS = dict([
    ("yellow_cone", (255, 255, 0)),
    ("blue_cone", (0, 0, 255)),
    ("large_orange_cone", (255, 122, 0)),
    ("orange_cone", (255, 122, 0)),
    ("unknown_cone", (255, 255, 255))
])

ID2LABEL = dict([
    (0, "yellow_cone"),
    (2, "blue_cone"),
    (3, "large_orange_cone"),
    (1, "orange_cone"),
    (4, "unknown_cone")
])

LABEL2ID = dict()
for k, v in ID2LABEL.items():  
   LABEL2ID[v]=k


def objects_threshold_scores(bboxes: torch.Tensor, 
                             labels: torch.Tensor=None, 
                             scores: torch.Tensor=None,
                             threshold_score: float=0.3):
    
    bboxes_copy = copy.deepcopy(bboxes)
    labels_copy = copy.deepcopy(labels)
    scores_copy = copy.deepcopy(scores)

    bboxes = torch.Tensor([])
    labels, scores = list(), list()
    for i, score in enumerate(scores_copy):
        if score >= threshold_score:
            bboxes = torch.cat((bboxes, bboxes_copy[i].unsqueeze(dim=0)), dim=0)
            labels.append(labels_copy[i])
            scores.append(score)
    
    scores = torch.Tensor(scores)

    del bboxes_copy, labels_copy, scores_copy

    return bboxes, labels, scores


def show_image_with_objects(image: np.array, 
                            bboxes_: torch.Tensor, 
                            labels: torch.Tensor=None, 
                            scores: torch.Tensor=None,
                            depths_value: torch.Tensor=None,
                            threshold_score: float=0.3):
    
    if image.shape[2] > 3:
        image = image.transpose(1, 2, 0)

    image = Image.fromarray(image)
    if scores != None:
        bboxes_, labels, scores = objects_threshold_scores(bboxes_, labels, scores, threshold_score)
        
    for i in range(len(bboxes_)):
        bboxes = bboxes_[i].flatten()
        draw = ImageDraw.Draw(image)
    
        if type(labels[i]) == str:
            draw.rectangle(bboxes.numpy(), outline = COLORS[labels[i]], width=2)
        else:
            draw.rectangle(bboxes.numpy(), outline = COLORS[ID2LABEL[int(labels[i])]], width=2)

        if scores != None:
            # TODO
            if depths_value != None:
                # TODO: scores
                # bbox = draw.textbbox((bboxes[0], bboxes[1]), f"{ID2LABEL[int(labels[i])]}\n distance: {0}")
                # draw.rectangle((bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2), fill=(0, 0, 0))
                if type(labels[i]) != str:
                    draw.text((bboxes[0], bboxes[1]-40), 
                            f"{ID2LABEL[int(labels[i])]}\n distance: {depths_value[i]:.2f}\n confidence: {scores[i]:.2f}", 
                            COLORS[ID2LABEL[int(labels[i])]])
                else:
                    draw.text((bboxes[0], bboxes[1]-40), 
                            f"{labels[i]}\n distance: {depths_value[i]:.2f}\n confidence: {scores[i]:.2f}", 
                            COLORS[labels[i]])
            else:
                if type(labels[i]) != str:
                    draw.text((bboxes[0], bboxes[1]-25), 
                              f"{ID2LABEL[int(labels[i])]}\n confidence: {scores[i]:.2f}", 
                              COLORS[ID2LABEL[int(labels[i])]])
                else:
                    print("WTF")
                    draw.text((bboxes[0], bboxes[1]-25), 
                              f"{labels[i]}\n confidence: {scores[i]:.2f}", 
                              COLORS[labels[i]])
        else:
            if depths_value != None:
                # TODO: scores
                # bbox = draw.textbbox((bboxes[0], bboxes[1]), f"{ID2LABEL[int(labels[i])]}\n distance: {0}")
                # draw.rectangle((bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2), fill=(0, 0, 0))
                draw.text((bboxes[0], bboxes[1]-40), 
                          f"{ID2LABEL[int(labels[i])]}\n distance: {depths_value[i]:.2f}", 
                          COLORS[ID2LABEL[int(labels[i])]])
            else:
                if type(labels[i]) != str:
                    draw.text((bboxes[0], bboxes[1]-15), 
                              f"{ID2LABEL[int(labels[i])]}", 
                              COLORS[ID2LABEL[int(labels[i])]])
                else:
                    draw.text((bboxes[0], bboxes[1]-15), 
                              f"{labels[i]}", 
                              COLORS[labels[i]])
    return image


def get_mono_640x192_model() -> List[networks]:
    model_name = "mono_640x192"

    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()

    return encoder, depth_decoder, loaded_dict_enc


def prediction(image: torch.Tensor, 
               model, 
               encoder, 
               depth_decoder,
               loaded_dict_enc,
               threshold_score=0.3):
    
    input_image = Image.fromarray(image.transpose(1, 2, 0))
    original_width, original_height = input_image.size

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_image_resized = input_image.resize((feed_width, feed_height), Image.LANCZOS)

    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)


    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]


    disp_resized = torch.nn.functional.interpolate(disp,
        (original_height, original_width), mode="bilinear", align_corners=False)

    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)

    def get_cone_distances(detected_cones, depth_map):
        distances = []
        
        for bbox in detected_cones:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            depth_crop = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(depth_crop)
            distances.append(avg_depth)
        
        return distances

    bboxes, labels, scores = model.predict(image, confidence=30, overlap=30)

    res = get_cone_distances(bboxes, 1 / disp_resized_np)

    return disp_resized_np, show_image_with_objects(image, 
                                                    bboxes, 
                                                    labels, 
                                                    scores, 
                                                    depths_value=res, 
                                                    threshold_score=threshold_score)


def process_frame(frame, 
                  model,
                  encoder,
                  depth_decoder,
                  loaded_dict_enc):
    
    disp_resized_np, annotated_frame = prediction(np.array(frame).transpose(2, 0, 1)[:3],
                                                  model,
                                                  encoder,
                                                  depth_decoder,
                                                  loaded_dict_enc)

    return disp_resized_np, np.array(annotated_frame)


def calculate_fps(start_time, num_frames):
    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time

    return fps