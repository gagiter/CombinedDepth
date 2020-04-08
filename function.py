
import torch
import libwarp
from torch.utils.tensorboard import SummaryWriter


class warper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, sample, occlusion=None):
        if occlusion is None:
            warped = libwarp.forward(image, sample)
            ctx.save_for_backward(image, sample, occlusion)
            return warped
        else:
            warped = libwarp.forward_width_occlusion(image, sample)
            ctx.save_for_backward(image, sample, occlusion)
            return warped

    @staticmethod
    def backward(ctx, grad_output):
        image, sample, occlusion = ctx.saved_tensors
        grad_image = grad_sample = grad_occlusion = None
        if occlusion is None and ctx.needs_input_grad[1]:
            grad_sample = libwarp.backward(image, sample, grad_output)

        return grad_image, grad_sample, grad_occlusion


if __name__ == '__main__':

    writer = SummaryWriter()

    image = torch.zeros([2, 3, 128, 256], dtype=torch.float32, device='cuda:0')
    ref = torch.zeros([2, 3, 128, 256], dtype=torch.float32, device='cuda:0')

    image[:, :, 30:80, 100:150] = 1.0
    ref[:, :, 20:70, 120:170] = 1.0

    grid_x = torch.linspace(-1.0, 1.0, 256)
    grid_y = torch.linspace(-1.0, 1.0, 128)

    grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack([grid_x, grid_y], dim=0)
    grid = grid[None, ...]
    grid = grid.repeat([2, 1, 1, 1])
    grid = grid.to('cuda:0')

    warp_0 = warper.apply(ref, grid)
    residual_ref = (image - ref).abs()
    grid_v = torch.cat([grid * 0.5 + 0.5, torch.ones([2, 1, 128, 256], device='cuda:0')], dim=1)

    writer.add_images('image/image', image, 0)
    writer.add_images('image/ref', ref, 0)
    writer.add_images('image/warp_0', warp_0, 0)
    writer.add_images('image/residual_ref', residual_ref, 0)
    writer.add_images('image/grid_v', grid_v, 0)

    move = torch.randn([2, 2, 1, 1], dtype=torch.float32, device='cuda:0')
    move *= 0.01
    move.requires_grad_()
    optimizer = torch.optim.SGD([move], 0.01)

    for i in range(200):
        sample = grid + move
        warp = warper.apply(ref, sample)
        residual_warp = (image - warp).abs()
        loss = residual_warp.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        writer.add_images('image/warp', warp, global_step=i)
        writer.add_images('image/residual_warp', residual_warp, global_step=i)
        writer.add_scalar('loss', loss, global_step=i)

    writer.close()
    # warped = libwarp.forward(image, sample)
    # help(libwarp)
    # help(libwarp.forward)