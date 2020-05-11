
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
import util

parser = argparse.ArgumentParser(description='CombinedDepth')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--encoder', type=str, default='mobilenet_v2')  #mobilenet_v2 resnet34
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--mini_batch_size', type=int, default=2)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='1')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--step_start', type=int, default=0)
parser.add_argument('--step_number', type=int, default=5000000)
parser.add_argument('--summary_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--ref_weight', type=float, default=1.0)
parser.add_argument('--depth_weight', type=float, default=1.0)
parser.add_argument('--regular_weight', type=float, default=1.0)
parser.add_argument('--ground_weight', type=float, default=1.0)
parser.add_argument('--scale_weight', type=float, default=1.0)
parser.add_argument('--depth_scale', type=float, default=1.0)
parser.add_argument('--rotation_scale', type=float, default=0.5)
parser.add_argument('--translation_scale', type=float, default=2.0)
parser.add_argument('--down_times', type=int, default=4)
parser.add_argument('--occlusion', type=int, default=1)
parser.add_argument('--use_number', type=int, default=0)
parser.add_argument('--target_pixels', type=int, default=300000)
parser.add_argument('--target_width', type=int, default=640)
parser.add_argument('--target_height', type=int, default=480)
parser.add_argument('--resume', type=int, default=1)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def train():
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_data = dataset.Data(args.data_root, target_pixels=args.target_pixels,
                              target_width=args.target_width, target_height=args.target_height,
                              use_number=args.use_number, device=device)
    train_loader = DataLoader(train_data, batch_size=args.mini_batch_size, shuffle=True)

    model = Model(args.encoder, rotation_scale=args.rotation_scale,
                  translation_scale=args.translation_scale,
                  depth_scale=args.depth_scale)

    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
    criteria = Criteria(
        depth_weight=args.depth_weight,
        regular_weight=args.regular_weight,
        ref_weight=args.ref_weight,
        ground_weight=args.ground_weight,
        scale_weight=args.scale_weight,
        down_times=args.down_times,
        occlusion=True if args.occlusion > 0 else False,
    )
    pcd = o3d.geometry.PointCloud()

    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    load_dir = os.path.join('checkpoint', args.model_name)
    if args.resume > 0 and os.path.exists(load_dir):
        model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pth')))
        optimiser.load_state_dict(torch.load(os.path.join(load_dir, 'optimiser.pth')))
        if os.path.exists(os.path.join(load_dir, 'step.pth')):
            args.step_start = torch.load(os.path.join(load_dir, 'step.pth'))['step']

    writer = SummaryWriter(os.path.join('runs', date_time + args.model_name))
    writer.add_text('args', str(args), 0)
    model.train()
    losses = []
    # optimiser.zero_grad()
    data_iter = iter(train_loader)
    for step in range(args.step_start, args.step_start + args.step_number):
        try:
            data_in = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data_in = next(data_iter)
        # data_in = next(train_data)
        data_out = model(data_in)
        loss = criteria(data_in, data_out)
        # optimiser.zero_grad()
        loss.backward()
        # optimiser.step()
        losses.append(loss.item())
        if step % (args.batch_size / args.mini_batch_size) == 0:
            optimiser.step()
            optimiser.zero_grad()

        if step % args.summary_freq == 0:
            loss = sum(losses) / len(losses)
            print('step:%d loss:%f' % (step, loss))
            util.visualize(data_in)
            util.visualize(data_out)
            writer.add_scalar('loss', loss, global_step=step)
            writer.add_image('image/image', data_in['image'][0], global_step=step)
            writer.add_image('image/color_map', data_in['color_map'][0], global_step=step)
            writer.add_image('image/normal', data_out['normal_v'][0], global_step=step)
            writer.add_text('camera', str(data_out['camera'][0].data.cpu().numpy()), global_step=step)
            if 'depth_v' in data_in:
                writer.add_image('image/depth_in', data_in['depth'][0], global_step=step)
            if 'depth_v' in data_out:
                writer.add_image('image/depth_out', data_out['depth'][0], global_step=step)
            if 'ground' in data_out:
                writer.add_text('ground', str(data_out['ground'][0].data.cpu().numpy()), global_step=step)
            for key in data_out:
                if key.startswith('base_'):
                    writer.add_image('image/' + key, data_out[key][0], global_step=step)
                elif key.startswith('image_'):
                    writer.add_image('image/' + key, data_out[key][0], global_step=step)
                elif key.startswith('residual_'):
                    writer.add_image('residual/' + key, data_out[key][0], global_step=step)
                elif key.startswith('warp_'):
                    writer.add_image('warp/' + key, data_out[key][0], global_step=step)
                elif key.startswith('grad_'):
                    writer.add_image('grad/' + key, data_out[key][0], global_step=step)
                elif key.startswith('regular_'):
                    writer.add_image('regular/' + key, data_out[key][0], global_step=step)
                elif key.startswith('ground_'):
                    writer.add_image('ground/' + key, data_out[key][0], global_step=step)
                elif key.startswith('loss'):
                    writer.add_scalar('loss/' + key, data_out[key], global_step=step)
                elif key.startswith('eval_'):
                    writer.add_scalar('eval/' + key, data_out[key], global_step=step)
                elif key.startswith('motion'):
                    writer.add_text('motion/' + key, str(data_out[key][0].data.cpu().numpy()), global_step=step)
            losses = []

        if step % args.save_freq == 0:
            save_dir = os.path.join('checkpoint', args.model_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            torch.save(optimiser.state_dict(), os.path.join(save_dir, 'optimiser.pth'))
            torch.save({'step': step}, os.path.join(save_dir, 'step.pth'))

            points = data_out['points'][0].data.cpu().numpy()
            points = points.transpose(1, 2, 0).reshape(-1, 3)
            pcd.points = o3d.utility.Vector3dVector(points)
            colors = data_in['image'][0].data.cpu().numpy()
            colors = colors.transpose(1, 2, 0).reshape(-1, 3)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(os.path.join(save_dir, '%s-%010d.pcd' % (args.model_name, step)), pcd)
            print('saved to ' + save_dir)

    writer.close()


if __name__ == '__main__':
    train()


