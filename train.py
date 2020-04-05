
import argparse
import torch
from torch.utils.data import DataLoader
import dataset
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import os
from criteria import Criteria
from model import Model
import util

parser = argparse.ArgumentParser(description='CombinedDepth')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--encoder', type=str, default='resnet34') # mobilenet_v2
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--epoch_start', type=int, default=0)
parser.add_argument('--epoch_num', type=int, default=50000)
parser.add_argument('--summary_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=100)
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda
args.device = torch.device('cuda:%d' % args.gpu_id if use_cuda else 'cpu')


def train():

    train_data = dataset.Data(args, mode='train')
    # val_data = dataset.Data(args, mode='val')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # model = smp.Unet('resnet34', encoder_weights=None, activation='sigmoid')
    model = Model(args.encoder)
    model = model.to(args.device)

    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    writer = SummaryWriter()
    criteria = Criteria()

    save_dir = os.path.join('checkpoint', args.model_name)
    if args.resume and os.path.exists(save_dir):
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pth')))
        optimiser.load_state_dict(torch.load(os.path.join(save_dir, 'optimiser.pth')))
        args.epoch_start = torch.load(os.path.join(save_dir, 'epoch.pth'))['epoch']

    for epoch in range(args.epoch_start, args.epoch_start + args.epoch_num):
        model.train()
        train_losses = []
        for data_in in train_loader:
            data_out = model(data_in)
            loss = criteria(data_in, data_out)
            optimiser.zero_grad()
            loss.backward()
            train_losses.append(loss.item())
            optimiser.step()

        if epoch % args.summary_freq == 0:
            loss = sum(train_losses) / len(train_losses)
            print(epoch, loss)
            util.colorize(data_in)
            util.colorize(data_out)
            writer.add_scalar('loss', loss, global_step=epoch)
            writer.add_image('image/image', data_in['image'][0], global_step=epoch)
            writer.add_image('image/mask', data_in['mask'][0], global_step=epoch)
            writer.add_image('image/depth_in', data_in['depth_c'][0], global_step=epoch)
            writer.add_image('image/depth_out', data_out['depth_c'][0], global_step=epoch)
            # writer.add_image('image/label', label[0], global_step=epoch)
            # writer.add_image('image/depth', depth[0], global_step=epoch)
            # writer.add_image('image/predict', output[0], global_step=epoch)
            # writer.add_image('image/normal', normal[0], global_step=epoch)
            # writer.add_image('image/overlap', overlap[0], global_step=epoch)
            # writer.add_image('image/dist', dist[0], global_step=epoch)

        if epoch % args.save_freq == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join('checkpoint', args.model_name, 'model.pth'))
            torch.save(optimiser.state_dict(), os.path.join('checkpoint', args.model_name, 'optimiser.pth'))
            torch.save({'epoch': epoch}, os.path.join('checkpoint', args.model_name, 'epoch.pth'))
            # torch.save(train_losses, 'train_losses.pth')

    writer.close()


if __name__ == '__main__':
    train()
