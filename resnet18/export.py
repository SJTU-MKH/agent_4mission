from operator import mod
import torch
import torchvision
import torch.nn as nn
import json
from torch.utils.mobile_optimizer import optimize_for_mobile

device = torch.device("cpu")
resnet18 = torchvision.models.resnet18(pretrained=False)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 8)

model_path = "./runs/model2.0/best_model.pth"
resnet18.load_state_dict(torch.load(model_path))

resnet18.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(resnet18, example, strict=False)

d = {"shape": example.shape}
extra_files = {'config.txt': json.dumps(d)}
f = "color.torchscript.ptl"
optimize_for_mobile(traced_script_module)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
