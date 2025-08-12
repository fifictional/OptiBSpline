import torch
import numpy as np
from acoustools.Utilities import device, DTYPE, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Levitator import LevitatorController
from vedo import Mesh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from acoustools.Visualiser import Visualise, ABC, get_image_positions
import numpy as np

def smooth_path(path_np, window_size=5):
    """
    Smooth a 3D path using moving average filter.

    Args:
      path_np: np.array of shape (N, 3)
      window_size: smoothing window size, odd number recommended

    Returns:
      smoothed_path: np.array of same shape (N, 3)
    """
    smoothed = np.zeros_like(path_np)
    for i in range(3):  # x, y, z
        smoothed[:, i] = np.convolve(path_np[:, i], 
                                     np.ones(window_size) / window_size, 
                                     mode='same')
    return smoothed


mesh = Mesh("stls/dumbells.stl")
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


# ------
from python_tsp.heuristics import solve_tsp_local_search
import numpy as np

dist_matrix = np.linalg.norm(control_points_np[:, None, :] - control_points_np[None, :, :], axis=2)
permutation, distance = solve_tsp_local_search(dist_matrix)
control_points_np = control_points_np[permutation]
# -------



# control_points_np = reorder_nearest(control_points_np)
control_points = torch.tensor(control_points_np, dtype=torch.float32, device=device)




n_interp = 3000
opt_curve = torch.nn.functional.interpolate(
    control_points.unsqueeze(0).transpose(1, 2),
    size=n_interp,
    mode='linear',
    align_corners=True
).transpose(1, 2).squeeze(0)

# ------
linear_path_np = opt_curve.cpu().numpy()
smoothed_linear_path_np = smooth_path(linear_path_np, window_size=15) 
opt_curve = torch.tensor(smoothed_linear_path_np, dtype=torch.float32, device=opt_curve.device)
# ------


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(control_points[:, 0].cpu(), control_points[:, 1].cpu(), control_points[:, 2].cpu(),
           color='red', label='Control Points')

ax.plot(opt_curve[:, 0].cpu(), opt_curve[:, 1].cpu(), opt_curve[:, 2].cpu(),
        color='green', label='Linear Interpolated Path')

ax.legend()
ax.set_title("Linear Interpolated Path and Control Points")
plt.show()


n_line_samples = 1000 
start_point = torch.tensor([[0.0, 0.0, 0.0]], device=opt_curve.device)
first_spline_point = opt_curve[0].unsqueeze(0)
line_points = torch.linspace(0, 1, n_line_samples, device=opt_curve.device).unsqueeze(1)
line_samples = start_point * (1 - line_points) + first_spline_point * line_points  
new_opt_curve = torch.cat([line_samples, opt_curve], dim=0)
print(new_opt_curve.norm(dim=1).diff())


lev = LevitatorController(ids=(73, 53)) 
# lev = LevitatorController(ids=(-1)) 
print("connected!")
lev.set_frame_rate(300)

xs = []

for point in new_opt_curve:
    p = point.unsqueeze(1).unsqueeze(0).to(device).to(DTYPE)/1200
    # print(f'position: {p}')
    wgs_out = wgs(p)
    activation = add_lev_sig(wgs_out)
    xs.append(activation)


lev.levitate(xs[0])

p0 = new_opt_curve[1].unsqueeze(1).unsqueeze(0).to(device).to(DTYPE)/1200
print(f'position: {p0}')
wgs_out0 = wgs(p0)
A, B, C = ABC(0.09, plane='xz')
Visualise(A, B, C, wgs_out0, points=p0)


input()
lev.levitate(xs)
lev.disconnect()

vel = opt_curve[1:] - opt_curve[:-1]
speeds = vel.norm(dim=1).cpu().numpy()
plt.plot(speeds)
plt.title("Velocity magnitude along linear path")
plt.show()
