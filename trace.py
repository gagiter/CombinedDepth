
import torch
import argparse
from model import Model
import os
import segmentation_models_pytorch as smp


model = Model()

example = torch.rand(1, 3, 512, 512)
traced_script_module = torch.jit.trace(model.depth_net, example)
traced_script_module.save('trace_model.pt')
