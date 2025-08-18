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


def compute_path_metrics(path, frame_rate):
    path_np = path.detach().cpu().numpy()
    diffs = np.diff(path_np, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)

    total_length = step_lengths.sum()
    num_frames = len(path_np)
    total_time = num_frames / frame_rate
    avg_speed = total_length / total_time

    return {
        "total_length": total_length,
        "total_time": total_time,
        "avg_speed": avg_speed,
        "step_lengths": step_lengths,
    }

def smooth_path(path_np, window_size=5):
    smoothed = np.zeros_like(path_np)
    for i in range(3):  
        smoothed[:, i] = np.convolve(path_np[:, i],
                                     np.ones(window_size) / window_size,
                                     mode='same')
    return smoothed

mesh = Mesh("stls/four-balls.stl")
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

n_interp = 3500
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
# torch.save(opt_curve.cpu(), 'linear_dumbells.pt')
# opt_curve = torch.load('linear_sphere.pt').to(device)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(*zip(*[p.cpu().numpy() for p in control_points]),
#            color='red', label='Control Points')
# ax.plot(opt_curve[:, 0].cpu(), opt_curve[:, 1].cpu(), opt_curve[:, 2].cpu(),
#         color='green', label='Interpolated Path')
# ax.legend()
# ax.set_title("Interpolated Path and Control Points")
# plt.show()

n_line_samples = 1500 
start_point = torch.tensor([[0.0, 0.0, 0.0]], device=opt_curve.device)
first_spline_point = opt_curve[0].unsqueeze(0)
line_points = torch.linspace(0, 1, n_line_samples, device=opt_curve.device).unsqueeze(1)
ease = (1 - torch.cos(line_points * np.pi)) / 2  
line_samples = start_point * (1 - ease) + first_spline_point * ease
new_opt_curve = torch.cat([line_samples, opt_curve], dim=0)
smoothed_np = smooth_path(new_opt_curve.cpu().numpy(), window_size=15)
new_opt_curve = torch.tensor(smoothed_np, dtype=torch.float32, device=opt_curve.device)
print(new_opt_curve.norm(dim=1).diff())


lev = LevitatorController(ids=(999, 1000)) 
# lev = LevitatorController(ids=(-1)) 
print("connected!")
lev.set_frame_rate(700)

xs = []

for point in new_opt_curve:
    p = point.unsqueeze(1).unsqueeze(0).to(device).to(DTYPE)/1400
    # print(f'position: {p}')
    wgs_out = wgs(p)
    activation = add_lev_sig(wgs_out)
    xs.append(activation)


lev.levitate(xs[0])

p0 = new_opt_curve[1].unsqueeze(1).unsqueeze(0).to(device).to(DTYPE)/1400
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


metrics = compute_path_metrics(new_opt_curve, frame_rate=700)
print(f"Path length: {metrics['total_length']:.3f} mm")
print(f"Path time: {metrics['total_time']:.3f} s")
print(f"Average speed: {metrics['avg_speed']:.3f} mm/s")
