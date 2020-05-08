import torch
from model import Model
import argparse
import os

parser = argparse.ArgumentParser(description='CombinedDepth')
parser.add_argument('--model_name', type=str, default='model_name')
parser.add_argument('--save_depth_name', type=str, default='model_depth.pt')
parser.add_argument('--save_camera_name', type=str, default='model_camera.pt')
args = parser.parse_args()

checkpoint_name = os.path.join('checkpoint', args.model_name,  'model.pth')
model = Model()
model.load_state_dict(torch.load(checkpoint_name))
example = torch.rand(1, 3, 512, 512)
traced_script_module_depth = torch.jit.trace(model.depth_net, example)
traced_script_module_depth.save(args.save_depth_name)
traced_script_module_camera = torch.jit.trace(model.camera_net, example)
traced_script_module_camera.save(args.save_camera_name)
