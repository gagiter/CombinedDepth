
import argparse
import torch
from torch.utils.data import DataLoader
import dataset
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import os
from criteria import Criteria

parser = argparse.ArgumentParser(description='CombinedDepth')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--epoch_start', type=int, default=0)
parser.add_argument('--epoch_num', type=int, default=50000)
parser.add_argument('--summary_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=10)
args = parser.parse_args()


def train():

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda:%d' % args.gpu_id if use_cuda else 'cpu')

    train_data = dataset.Data(args)
    val_data = dataset.Data(args)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model = smp.Unet('resnet34', encoder_weights=None, activation='sigmoid')
    model = model.to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    writer = SummaryWriter()
    criteria = Criteria()

    if args.resume:
        model.load_state_dict(torch.load(os.path.join('checkpoint', args.dataset, 'model.pth')))
        optimiser.load_state_dict(torch.load(os.path.join('checkpoint', args.dataset, 'optimiser.pth')))
        args.epoch_start = torch.load(os.path.join('checkpoint', args.dataset, 'epoch.pth'))['epoch']

    for epoch in range(args.epoch_start, args.epoch_start + args.epoch_num):
        model.train()
        train_losses = []
        for data in train_loader:
            predict = model(data)
            loss = criteria(predict, data)
            optimiser.zero_grad()
            loss.backward()
            train_losses.append(loss.item())
            optimiser.step()

        if epoch % args.summary_freq == 0:
            loss = sum(train_losses) / len(train_losses)
            print(epoch, loss)
            writer.add_scalar('loss', loss, global_step=epoch)
            # writer.add_image('image/color', color[0], global_step=epoch)
            # writer.add_image('image/label', label[0], global_step=epoch)
            # writer.add_image('image/depth', depth[0], global_step=epoch)
            # writer.add_image('image/predict', output[0], global_step=epoch)
            # writer.add_image('image/normal', normal[0], global_step=epoch)
            # writer.add_image('image/overlap', overlap[0], global_step=epoch)
            # writer.add_image('image/dist', dist[0], global_step=epoch)

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join('checkpoint', args.dataset, 'model.pth'))
            torch.save(optimiser.state_dict(), os.path.join('checkpoint', args.dataset, 'optimiser.pth'))
            torch.save({'epoch': epoch}, os.path.join('checkpoint', args.dataset, 'epoch.pth'))
            # torch.save(train_losses, 'train_losses.pth')
    print('train')
    pass


if __name__ == '__main__':
    train()
