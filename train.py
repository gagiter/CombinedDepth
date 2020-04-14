
import argparse
import torch
from torch.utils.data import DataLoader
import dataset
from torch.utils.tensorboard import SummaryWriter
import os
from criteria import Criteria
from model import Model
import open3d as o3d
from datetime import datetime

parser = argparse.ArgumentParser(description='CombinedDepth')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--encoder', type=str, default='resnet34')  # mobilenet_v2
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--load_name', type=str, default='model')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--mini_batch_size', type=int, default=2)
parser.add_argument('--resume', type=int, default=1)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='1')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--epoch_start', type=int, default=0)
parser.add_argument('--epoch_num', type=int, default=50000)
parser.add_argument('--summary_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--plane_weight', type=float, default=1.0)
parser.add_argument('--normal_weight', type=float, default=1.0)
parser.add_argument('--ref_weight', type=float, default=1.0)
parser.add_argument('--depth_weight', type=float, default=1.0)
parser.add_argument('--depth_scale', type=float, default=1.0)
parser.add_argument('--rotation_scale', type=float, default=0.5)
parser.add_argument('--translation_scale', type=float, default=2.0)
parser.add_argument('--down_times', type=int, default=4)
parser.add_argument('--occlusion', type=int, default=1)
parser.add_argument('--trace_jit', type=int, default=1)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def train():
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_data = dataset.Data(args.data_root, mode='train', device=device)
    # val_data = dataset.Data(args, mode='val')

    train_loader = DataLoader(train_data, batch_size=args.mini_batch_size, shuffle=True)
    model = Model(args.encoder, rotation_scale=args.rotation_scale,
                  translation_scale=args.translation_scale,
                  depth_scale=args.depth_scale)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
    writer = SummaryWriter(os.path.join('runs', date_time + args.model_name))
    criteria = Criteria(
        depth_weight=args.depth_weight,
        normal_weight=args.normal_weight,
        plane_weight=args.plane_weight,
        ref_weight=args.ref_weight,
        down_times=args.down_times,
        occlusion=True if args.occlusion > 0 else False
    )
    pcd = o3d.geometry.PointCloud()
    load_dir = os.path.join('checkpoint', args.load_name)
    if args.resume > 0 and os.path.exists(load_dir):
        model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pth')))
        optimiser.load_state_dict(torch.load(os.path.join(load_dir, 'optimiser.pth')))
        args.epoch_start = torch.load(os.path.join(load_dir, 'epoch.pth'))['epoch']

    if args.trace_jit > 0:
        example = torch.rand(1, 3, 256, 512)
        traced_script_module = torch.jit.trace(model.depth_net, example)
        traced_script_module.save(os.path.join(load_dir, "trace_model.pt"))
    model = model.to(device)

    for epoch in range(args.epoch_start, args.epoch_start + args.epoch_num):
        model.train()
        optimiser.zero_grad()
        train_losses = []
        for i, data_in in enumerate(train_loader):
            data_out = model(data_in)
            loss = criteria(data_in, data_out)
            loss.backward()
            train_losses.append(loss.item())
            if (i + 1) % (args.batch_size // args.mini_batch_size) == 0:
                optimiser.step()
                optimiser.zero_grad()

            step = epoch * len(train_loader) + i
            if step % args.summary_freq == 0:
                loss = sum(train_losses) / len(train_losses)
                print('step:%d[epoch:%d idx:%d] loss:%f' % (step, epoch, i, loss))
                if 'abs_rel' in data_out:
                    writer.add_scalar('eval/abs_rel', data_out['abs_rel'], global_step=step)
                writer.add_scalar('loss', loss, global_step=step)
                writer.add_image('image/image', data_in['image'][0], global_step=step)
                writer.add_image('image/depth_in', data_in['depth'][0] * 1.0, global_step=step)
                writer.add_image('image/depth_out', data_out['depth'][0] * 1.0, global_step=step)
                writer.add_image('image/image_grad', data_out['image_grad'][0], global_step=step)
                for key in data_out:
                    if 'residual' in key:
                        writer.add_image('residual/' + key, data_out[key][0], global_step=step)
                    elif 'loss' in key:
                        writer.add_scalar('loss/' + key, data_out[key], global_step=step)
                    elif 'motion' in key:
                        writer.add_text('motion/' + key, str(data_out[key][0].data.cpu().numpy()), global_step=step)

            if step % args.save_freq == 0:
                save_dir = os.path.join('checkpoint', args.model_name)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
                torch.save(optimiser.state_dict(), os.path.join(save_dir, 'optimiser.pth'))
                torch.save({'epoch': epoch}, os.path.join(save_dir, 'epoch.pth'))

                points = data_out['points'][0].data.cpu().numpy()
                points = points.transpose(1, 2, 0).reshape(-1, 3)
                pcd.points = o3d.utility.Vector3dVector(points)
                colors = data_in['image'][0].data.cpu().numpy()
                colors = colors.transpose(1, 2, 0).reshape(-1, 3)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                normals = data_out['normals'][0].data.cpu().numpy()
                normals = normals.transpose(1, 2, 0).reshape(-1, 3)
                pcd.normals = o3d.utility.Vector3dVector(normals)
                o3d.io.write_point_cloud(os.path.join(save_dir, '%s-%010d.pcd' % (args.model_name, step)), pcd)
                print('saved to ' + save_dir)

    writer.close()


if __name__ == '__main__':
    train()


