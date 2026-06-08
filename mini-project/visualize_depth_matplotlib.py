import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parse the .ply file
def visualize_pointcloud(file_path, save_path='output/pointcloud_3d.png'):
    pts = []
    cols = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header_end = next(i for i, line in enumerate(lines) if line.strip() == 'end_header')
        for line in lines[header_end + 1:]:
            parts = line.strip().split()
            if len(parts) == 6:
                x, y, z, r, g, b = map(float, parts)
                pts.append([x, y, z])
                cols.append([r/255, g/255, b/255])

    pts = np.array(pts)
    cols = np.array(cols)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')