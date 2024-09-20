import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict

from utils import LABEL2ID, ID2LABEL



class YOLOv9(torch.nn.Module):
    def __init__(self, num_classes: int=4) -> None:
        super().__init__()

        self.model = YOLO("../weights/best.pt")

        self.LABEL2LABEL = dict([
            ("unknown_cone", "blue_cone"),
            ("large_orange_cone", "blue_cone"),
            ("yellow_cone", "yellow_cone"),
            ("blue_cone", "blue_cone"),
            ("orange_cone", "orange_cone")
        ])

    def predict(self, X: torch.Tensor, confidence=40, overlap=30) -> torch.Tensor:
        results = self.model.predict(X.transpose(1, 2, 0))
        bboxes = results[0].boxes.data[:, :4]
        labels_ = results[0].boxes.cls
        scores = results[0].boxes.conf
        labels = np.zeros_like(labels_.cpu())
        for i, label in enumerate(labels_):
            label = int(label.item())
            labels[i] = LABEL2ID[self.LABEL2LABEL[ID2LABEL[label]]]
            # print(label_, label)
            
        return bboxes.cpu(), labels, scores.cpu()

    
    # To calculate the loss function
    def forward(self, images: List[torch.Tensor], annotation: List[Dict[str, torch.Tensor]]) -> Dict[str, int]:
        return self.model(images, annotation)