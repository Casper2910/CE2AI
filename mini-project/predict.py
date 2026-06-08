import torch
import numpy as np
from PIL import Image

import sys
sys.path.append('/var/home/cbramm/Documents/AAU/CE2/CE2AI/mini-project')
from train import DepthTransformer, ConvDecoder, Block, MultiHeadAttention, Head, MLP, PatchEmbed

from visualize_depth_matplotlib import visualize_pointcloud


image_path = 'images/test_shapes.png'
model_path = 'output/depth_model.pth'
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
img_size   = 512

# load the whole model, no class definitions needed
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# load and preprocess image
pil_img        = Image.open(image_path).convert('RGB')
orig_w, orig_h = pil_img.size
rgb_np  = np.array(pil_img.resize((img_size, img_size)), dtype=np.float32) / 255.0
tensor  = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

# run inference
with torch.no_grad():
    depth = model(tensor)[0].squeeze().cpu().numpy()

# save depth image
Image.fromarray((depth * 255).astype(np.uint8)).save('output/shapes_depth.png')

# resize depth back to original image size for point cloud
depth_orig = np.array(
    Image.fromarray((depth * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR),
    dtype=np.float32
) / 255.0
rgb_orig = np.array(pil_img, dtype=np.uint8)

# build point cloud
fov_deg = 60.0
f       = (orig_w / 2) / np.tan(np.deg2rad(fov_deg / 2))
Z       = 10.0 - depth_orig * 9.5          # normalised depth → metric distance
yy, xx  = np.mgrid[0:orig_h, 0:orig_w]
X       = (xx - orig_w / 2) / f * Z
Y       = (yy - orig_h / 2) / f * Z

pts  = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
cols = rgb_orig.reshape(-1, 3)

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

with open('output/shapes_pointcloud.ply', 'w') as f_out:
    f_out.write(header + rows)

visualize_pointcloud('output/shapes_pointcloud.ply', 'output/shapes_pointcloud_3d.png')

print("saved output/shapes_depth.png")
print("saved output/shapes_pointcloud.ply")
print("saved output/shapes_pointcloud_3d.png")