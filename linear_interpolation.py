import torch
import numpy as np
from acoustools.Utilities import device, DTYPE, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Levitator import LevitatorController
from vedo import Mesh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mesh = Mesh("stls/two-balls.stl")
N = 500
indices = np.random.choice(mesh.npoints, N, replace=False)
sampled = mesh.clone().triangulate()
sampled.points = sampled.points[indices]
control_points_np = sampled.points

pca = PCA(n_components=1)
proj = pca.fit_transform(control_points_np)
sorted_idx = proj[:, 0].argsort()
control_points_np = control_points_np[sorted_idx]

def reorder_nearest(points_np):
    reordered = [points_np[0]]
    points_np = np.delete(points_np, 0, axis=0)
    while len(points_np) > 0:
        dists = np.linalg.norm(points_np - reordered[-1], axis=1)
        idx = np.argmin(dists)
        reordered.append(points_np[idx])
        points_np = np.delete(points_np, idx, axis=0)
    return np.array(reordered)

control_points_np = reorder_nearest(control_points_np)

control_points = torch.tensor(control_points_np, dtype=torch.float32, device=device)

n_interp = 2000
linear_path = torch.nn.functional.interpolate(
    control_points.unsqueeze(0).transpose(1, 2),
    size=n_interp,
    mode='linear',
    align_corners=True
).transpose(1, 2).squeeze(0)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(control_points[:, 0].cpu(), control_points[:, 1].cpu(), control_points[:, 2].cpu(),
           color='red', label='Control Points')

ax.plot(linear_path[:, 0].cpu(), linear_path[:, 1].cpu(), linear_path[:, 2].cpu(),
        color='green', label='Linear Interpolated Path')

ax.legend()
ax.set_title("Linear Interpolated Path and Control Points")
plt.show()

# # Calculate WGS activations
# xs_linear = []
# for point in linear_path:
#     p = point.unsqueeze(1).unsqueeze(0).to(device).to(DTYPE) / 1000
#     wgs_out = wgs(p)
#     activation = add_lev_sig(wgs_out)
#     xs_linear.append(activation)

