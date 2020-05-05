import torch
from model import Model


model = Model()
model.load_state_dict(torch.load('checkpoint/normal_data/model.pth'))
example = torch.rand(1, 3, 512, 512)
traced_script_module_depth = torch.jit.trace(model.depth_net, example)
traced_script_module_depth.save('model_depth.pt')
traced_script_module_camera = torch.jit.trace(model.camera_net, example)
traced_script_module_camera.save('model_camera.pt')