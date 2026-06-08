import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from visualize_depth_matplotlib import visualize_pointcloud

class SyntheticDepthDataset(Dataset):

    def __init__(self, length, img_size):
        self.length = length
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        H = W = self.img_size
        rgb   = np.zeros((H, W, 3), dtype=np.float32)
        depth = np.zeros((H, W),    dtype=np.float32)

        shapes = [self._random_shape(H, W) for _ in range(random.randint(3, 8))]
        shapes.sort(key=lambda s: s['depth'])

        for s in shapes:
            rgb[s['mask']]   = s['colour']
            depth[s['mask']] = s['depth']

        yy, xx   = np.mgrid[0:H, 0:W] / max(H, W)
        bg_color = np.stack([xx, yy, 1 - xx], axis=-1).astype(np.float32) * 0.15
        bg_mask  = depth == 0
        rgb[bg_mask]   = bg_color[bg_mask]
        depth[bg_mask] = 0.05

        rgb_t   = torch.from_numpy(rgb.transpose(2, 0, 1))   # (3, H, W)
        depth_t = torch.from_numpy(depth).unsqueeze(0)        # (1, H, W)
        return rgb_t, depth_t

    def _random_shape(self, H, W):
        cx  = random.randint(0, W - 1)
        cy  = random.randint(0, H - 1)
        rw  = random.randint(W // 16, W // 3)
        rh  = random.randint(H // 16, H // 3)
        col = np.array([random.random(), random.random(), random.random()],
                       dtype=np.float32)
        d   = random.uniform(0.1, 1.0)
        yy, xx = np.ogrid[0:H, 0:W]
        if random.random() < 0.5:
            mask = (xx >= cx-rw) & (xx < cx+rw) & (yy >= cy-rh) & (yy < cy+rh)
        else:
            mask = (xx-cx)**2/rw**2 + (yy-cy)**2/rh**2 <= 1.0
        return {'mask': mask, 'colour': col, 'depth': d}


# Example usage: generate one sample and save as .ply and .png
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    # get one sample from the dataset
    dataset   = SyntheticDepthDataset(length=1, img_size=512)
    rgb_t, depth_t = dataset[0]

    # convert tensors back to numpy
    rgb_np   = rgb_t.permute(1, 2, 0).numpy()           # (H, W, 3)
    depth_np = depth_t.squeeze(0).numpy()               # (H, W)

    rgb_u8   = (rgb_np   * 255).astype(np.uint8)
    depth_u8 = (depth_np * 255).astype(np.uint8)

    # save 2d image
    Image.fromarray(rgb_u8).save('data/sample_rgb.png')
    Image.fromarray(depth_u8).save('data/sample_depth.png')
    print('saved data/sample_rgb.png')
    print('saved data/sample_depth.png')

    # build point cloud
    H, W     = depth_np.shape
    fov_deg  = 60.0
    f        = (W / 2) / np.tan(np.deg2rad(fov_deg / 2))
    Z        = 10.0 - depth_np * 9.5          # normalised depth → metric distance
    yy, xx   = np.mgrid[0:H, 0:W]
    X        = (xx - W / 2) / f * Z
    Y        = (yy - H / 2) / f * Z

    pts  = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    cols = rgb_u8.reshape(-1, 3)

    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rows = "\n".join(
        f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}"
        for p, c in zip(pts, cols)
    )

    with open('data/sample_pointcloud.ply', 'w') as f_out:
        f_out.write(header + rows)
    print('saved data/sample_pointcloud.ply')

    visualize_pointcloud('data/sample_pointcloud.ply', 'data/sample_pointcloud_3d.png')
    print('saved data/sample_pointcloud_3d.png')