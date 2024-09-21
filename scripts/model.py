import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict



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


class YOLOv9(torch.nn.Module):
    def __init__(self, path2weights: Path, num_classes: int=4) -> None:
        super().__init__()

        self.path2weights = path2weights
        self.model = YOLO(f"{path2weights}")

        self.LABEL2LABEL = dict([
            ("unknown_cone", "blue_cone"),
            ("large_orange_cone", "blue_cone"),
            ("yellow_cone", "yellow_cone"),
            ("blue_cone", "blue_cone"),
            ("orange_cone", "orange_cone")
        ])

    def predict(self, X: torch.Tensor, confidence=40, overlap=30) -> torch.Tensor:

        if self.path2weights[-2] != "pt":
            results = self.model.predict(source=X.transpose(1, 2, 0), device=0)
        else:
            results = self.model.predict(source=X.transpose(1, 2, 0))

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
    
    def export2onnx(self, frame_size=640):
        self.model.export(format="onnx", 
                          imgsz = frame_size,
                          )
        
        return self.model