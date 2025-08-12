import torch
import numpy as np
from acoustools.Utilities import device, DTYPE, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Levitator import LevitatorController
from vedo import Mesh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from acoustools.Visualiser import Visualise, ABC, get_image_positions
from acoustools.Paths.Interpolate import interpolate_path  

def smooth_path(path_np, window_size=5):
    smoothed = np.zeros_like(path_np)
    for i in range(3):  
        smoothed[:, i] = np.convolve(path_np[:, i],
                                     np.ones(window_size) / window_size,
                                     mode='same')
    return smoothed

mesh = Mesh("stls/sphere.stl")
N = 500
indices = np.random.choice(mesh.npoints, N, replace=False)
sampled = mesh.clone().triangulate()
sampled.points = sampled.points[indices]
control_points_np = sampled.points

pca = PCA(n_components=1)
proj = pca.fit_transform(control_points_np)
sorted_idx = proj[:, 0].argsort()
control_points_np = control_points_np[sorted_idx]

from python_tsp.heuristics import solve_tsp_local_search
dist_matrix = np.linalg.norm(control_points_np[:, None, :] - control_points_np[None, :, :], axis=2)
permutation, distance = solve_tsp_local_search(dist_matrix)
control_points_np = control_points_np[permutation]

control_points = [torch.tensor(pt, dtype=torch.float32, device=device) for pt in control_points_np]

n_interp = 3000
opt_curve_list = interpolate_path(control_points, n_interp)

# Debug: print shapes
for i, p in enumerate(opt_curve_list):
    print(f"{i}: shape={p.shape}")

# Filter out invalid points
opt_curve_list = [p for p in opt_curve_list if p.numel() == 3]

opt_curve = torch.stack(opt_curve_list, dim=0)


linear_path_np = opt_curve.cpu().numpy()
smoothed_linear_path_np = smooth_path(linear_path_np, window_size=15)
opt_curve = torch.tensor(smoothed_linear_path_np, dtype=torch.float32, device=opt_curve.device)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*zip(*[p.cpu().numpy() for p in control_points]),
           color='red', label='Control Points')
ax.plot(opt_curve[:, 0].cpu(), opt_curve[:, 1].cpu(), opt_curve[:, 2].cpu(),
        color='green', label='Interpolated Path')
ax.legend()
ax.set_title("Interpolated Path and Control Points")
plt.show()
