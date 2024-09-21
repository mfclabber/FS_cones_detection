import onnx

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import YOLOv9


if __name__ == "__main__":
    
    torch_model = YOLOv9()

    # torch_input = torch.randn(1, 1, 1280, 640)
    # onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)

    # onnx_program.save("../weights/yolov9t.onnx")

    torch_model.export2onnx()
    # onnx_model = onnx.load("/home/mfclabber/fs_cones_detection&monodepth/weights/best.onnx")
    # onnx.checker.check_model(onnx_model)