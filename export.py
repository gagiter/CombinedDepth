import torch
from model import Model


model = Model()
model.load_state_dict(torch.load('checkpoint/model_depth_03/model.pth'))
example = torch.rand(1, 3, 512, 512)
traced_script_module = torch.jit.trace(model.depth_net, example)
traced_script_module.save('trace_model.pt')
