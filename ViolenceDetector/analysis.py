from ultralytics import YOLO
from thop import profile
import torch

# load model
yolo = YOLO("best.pt")

# extract the pytorch model
model = yolo.model

model.eval()

dummy = torch.randn(1,3,320,320)

flops, params = profile(model, inputs=(dummy,))

print("FLOPs:", flops)
print("Params:", params)

import onnx
from onnx_tool import model_profile

model = onnx.load("gapconv1d.onnx")

profile = model_profile(model)

print(profile)