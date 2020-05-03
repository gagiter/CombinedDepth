import argparse
import torch
from model import Model
import os
import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time


parser = argparse.ArgumentParser(description='CombinedDepth')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--encoder', type=str, default='mobilenet_v2')
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='1')
parser.add_argument('--target_pixels', type=int, default=300000)
parser.add_argument('--target_width', type=int, default=640)
parser.add_argument('--target_height', type=int, default=480)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')


data = dataset.Data(args.data_root, target_pixels=args.target_pixels,
                    target_width=args.target_width, target_height=args.target_height,
                    device=device)
loader = DataLoader(data, batch_size=1, shuffle=False)


model = Model(args.encoder)
model = model.to(device)
load_dir = os.path.join('checkpoint', args.model_name)
model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pth')))
model.eval()

date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
writer = SummaryWriter(os.path.join('test', date_time + args.model_name))
writer.add_text('args', str(args), 0)

with torch.no_grad():
    for idx, data_in in enumerate(loader):
        start = time.time()
        depth = model.depth_net(data_in['image'])
        elapsed = time.time() - start
        writer.add_image('image', data_in['image'][0], global_step=idx)
        writer.add_image('depth', depth[0], global_step=idx)
        print(idx, len(loader), elapsed)
        # print('aaa%d' % idx, end='\r')
        # print('[{0}] {1}%'.format('#' * 5, idx))

