import torch
from model import Model


model = Model()
model.load_state_dict(torch.load('checkpoint/global_depth/model.pth'))
example = torch.rand(1, 3, 512, 512)
traced_script_module = torch.jit.trace(model.depth_net, example)
traced_script_module.save('model_depth.pt')
