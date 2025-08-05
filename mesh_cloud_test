import torch
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from OptiBSpline import evaluate_bspline, OptiBSpline
import json
import os

# from acoustools.Gorkov import gorkov_autograd, gorkov_analytical
from my_Gorkov import gorkov_analytical, get_gorkov_constants
from acoustools.Utilities import TRANSDUCERS, device, DTYPE, create_points, propagate_abs, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, ABC, get_image_positions

from vedo import Mesh, show, Points
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
import numpy as np

import time 

import torch

# def load_control_points(file_path="control_points2.json"):
#     desktop = os.path.join(os.path.expanduser("~"), "Desktop", "MSD")
#     full_path = os.path.join(desktop, file_path)
    
#     with open(full_path, 'r') as f:
#         points = json.load(f)
    
#     return torch.tensor(points, dtype=torch.float32)


# def gorkov_with_gradients(activations, points, target_pressure=5000.0):
#     """
#     Properly integrates gorkov_analytical with gradient flow
#     Args:
#         activations: Complex activations [B x N_trans x 1]
#         points: Evaluation points [B x 3 x N]
#         target_pressure: Desired max pressure in Pascals
#     Returns:
#         U: Gorkov potential tensor
#     """
#     # Ensure gradient connection
#     points = points.clone().requires_grad_(True)
    
#     # Compute scaling factor (detached from graph)
#     with torch.no_grad():
#         pressure = propagate_abs(activations, points)
#         scale = target_pressure / (pressure.abs().max() + 1e-10)
    
#     # Compute scaled Gorkov potential
#     U = gorkov_analytical(activations * scale, points)
    
#     # Manually connect gradients
#     grad_output = torch.ones_like(U)
#     torch.autograd.grad(
#         outputs=U,
#         inputs=points,
#         grad_outputs=grad_output,
#         create_graph=True,  # For higher-order derivatives
#         retain_graph=True   # Keep graph for multiple backward passes
#     )
    
#     return U

def bspline_objective2(control_points, target_points, degree, knot_vector,
                     n_samples=600, alpha=10.0, beta=10.0, gamma=500000.0, delta=1.0):
    control_points = control_points.to(device).to(DTYPE)
    control_points = control_points.clone().requires_grad_(True)
    
    t_values = torch.linspace(knot_vector[degree], knot_vector[-degree-1], n_samples,
                            device=control_points.device)
    curve_points = evaluate_bspline(control_points, knot_vector, degree, t_values)
    curve_points.retain_grad()
    
    # Smoothness loss
    with torch.no_grad():
        vel = curve_points[1:] - curve_points[:-1]
        total_length = vel.norm(dim=1).sum()
        target_length = total_length / vel.shape[0]
        target_length = (vel.norm(dim=1).sum() / vel.shape[0]).detach()

    # vel = curve_points[1:] - curve_points[:-1]
    loss_velocity = (vel.norm(dim=1) - target_length).square().sum()  # Penalize length variations
    accel = vel[1:] - vel[:-1]
    loss_accel = (accel**2).sum()
    loss_smooth = 0.1*loss_velocity + loss_accel  
    
    # Fit loss
    loss_fit = ((curve_points - target_points)**2).sum() if target_points is not None \
              else torch.tensor(0.0, device=device)
    
    # Gorkov potential unbatched 
    # loss_gorkov = torch.tensor(0.0, device=device, dtype=DTYPE)

    # curve_points = curve_points.to(device).to(DTYPE)


    # if gamma > 0:
    #     batch_size = 20  
    #     for i in range(0, len(curve_points), batch_size):
    #         batch = curve_points[i:i+batch_size]                     
    #         p_batch = batch.T.unsqueeze(0)                          

    #         wgs_out = wgs(p_batch)                   
    #         act = add_lev_sig(wgs_out) 

    #         act_abs = act.abs()
    #         act_min = act_abs.min().item()
    #         act_max = act_abs.max().item()

    #         # target_pressure_amplitude = 1e3 
    #         # act = act * (target_pressure_amplitude / act.abs().max())


    #         g_vals = gorkov_analytical(act, p_batch)                
    #         loss_gorkov += g_vals.sum()
   
    gorkov_vals = []
    for point in curve_points:
        p = point.view(1, 3).T.unsqueeze(0).to(device).to(DTYPE)  
        wgs_out = wgs(p) 
        act = add_lev_sig(wgs_out) 
        target_pressure = 300
        act = act * (target_pressure / (act.abs().max() + 1e-10)) 
        g_val = gorkov_analytical(act, p)  
        gorkov_vals.append(g_val)

    gorkov_vals = torch.stack(gorkov_vals) 
    loss_gorkov = gorkov_vals.sum()  
    loss_gorkov = torch.abs(gorkov_vals).sum()  #  positive loss

    K1, K2 = get_gorkov_constants()
    print(f"K1: {K1}, K2: {K2}") 

    # activations = torch.ones((1, TRANSDUCERS.shape[0], 1), 
    #                dtype=torch.complex64,  # Must match propagator
    #                device=device)
    
    # points_t = curve_points.T.unsqueeze(0).to(torch.float32)
    # gorkov = unified_gorkov(activations, points_t, target_pressure=5000.0)
    # loss_gorkov = gorkov.sum()

    # points_t = curve_points.T.unsqueeze(0).to(torch.float64)
    # activations = torch.ones((1, len(TRANSDUCERS), 1),
    #                        dtype=torch.complex128,
    #                        device=device)
    
    # gorkov = gorkov_with_gradients(activations, points_t)
    # loss_gorkov = gorkov.sum()


    # Gorkov potential batched 
    # curve_points_batched = curve_points.T.unsqueeze(0)  
    # wgs_out = wgs(curve_points_batched)
    # act = add_lev_sig(wgs_out)
    # g_vals = gorkov_analytical(act, curve_points_batched) 

    # Curvature loss - penalises sharp bends in the curve
    vel = curve_points[1:] - curve_points[:-1]
    accel = vel[1:] - vel[:-1]
    curvature = (accel.norm(dim=1)**2) / (vel[:-1].norm(dim=1)**3 + 1e-6)
    loss_curvature = curvature.sum()

    # total_loss = alpha * loss_smooth + beta * loss_fit + delta * loss_curvature
    total_loss = alpha * loss_smooth + beta * loss_fit + gamma * loss_gorkov + delta * loss_curvature

    print(f"Smooth: {alpha * loss_smooth.item():.4f}, Fit: {beta * loss_fit.item():.4f}, Gorkov: {gamma * loss_gorkov.item():.4f}, Curv: {delta * loss_curvature.item():.4f}, Total: {total_loss.item():.4f}")

    return {
        "total": total_loss,
        "smoothness": loss_smooth,
        "fit": loss_fit,
        "gorkov": loss_gorkov,
        "curvature": loss_curvature,
    }


def bspline_objective(control_points, target_points, degree, knot_vector,
                     n_samples=600, alpha=100.0, beta=1.0, gamma=50.0, delta=1000.0):
    control_points = control_points.to(device).to(DTYPE)

    t_values = torch.linspace(knot_vector[degree], knot_vector[-degree - 1], n_samples, device=control_points.device)
    curve_points = evaluate_bspline(control_points, knot_vector, degree, t_values)

    # Smoothness loss
    # accel = curve_points[2:] - 2 * curve_points[1:-1] + curve_points[:-2]
    # loss_smooth = (accel**2).sum()

    # Smoothness loss
    with torch.no_grad():
        vel = curve_points[1:] - curve_points[:-1]
        total_length = vel.norm(dim=1).sum()
        target_length = total_length / vel.shape[0]
        target_length = (vel.norm(dim=1).sum() / vel.shape[0]).detach()

    vel = curve_points[1:] - curve_points[:-1]
    loss_velocity = (vel.norm(dim=1) - target_length).square().sum()  # Penalize length variations
    accel = vel[1:] - vel[:-1]
    loss_accel = (accel**2).sum()
    jerk = accel[1:] - accel[:-1]
    loss_jerk = jerk.pow(2).sum()
    loss_smooth = 0.1 * loss_velocity + loss_accel + 1 * loss_jerk


    # Fit loss
    loss_fit = ((curve_points - target_points)**2).sum() if target_points is not None else torch.tensor(0.0, device=device)

    # Curvature loss
    vel = curve_points[1:] - curve_points[:-1]
    accel = vel[1:] - vel[:-1]
    curvature = (accel.norm(dim=1)**2) / (vel[:-1].norm(dim=1)**3 + 1e-6)
    loss_curvature = curvature.sum()


    total_loss = alpha * loss_smooth + beta * loss_fit  + delta * loss_curvature
    print(f"Smooth: {alpha * loss_smooth.item():.4f}, Fit: {beta * loss_fit.item():.4f}, Curv: {delta * loss_curvature.item():.4f}, Total: {total_loss.item():.4f}")

    

    return {
        "total": total_loss,
        "smoothness": loss_smooth,
        "fit": loss_fit,
        "curvature": loss_curvature,
    }


# Knot vector generator
# def generate_knot_vector(n_control_points: int, degree: int) -> torch.Tensor:
#     n_knots = n_control_points + degree + 1
#     knots = [0.0] * (degree + 1)
#     knots += list(torch.linspace(0, 1, n_knots - 2 * (degree + 1)))
#     knots += [1.0] * (degree + 1)
#     return torch.tensor(knots, dtype=torch.float32, device=device)

def generate_knot_vector(control_points, degree):
    """Generate exact-length knot vector for given control points"""
    n = len(control_points)
    
    if isinstance(control_points, torch.Tensor):
        pts = control_points.cpu().numpy()
    else:
        pts = control_points
    
    # For chord-length parameterization:
    if n > degree + 1:
        # Compute chord lengths
        diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cum_length = np.concatenate([[0], np.cumsum(diffs)])
        interior = cum_length[1:-1] / cum_length[-1]  # Skip first/last
        
        # Distribute interior knots
        num_interior = n - degree - 1
        if num_interior > 0:
            interior_knots = np.linspace(0, 1, num_interior + 2)[1:-1]  # Evenly spaced
        else:
            interior_knots = np.array([])
    else:
        interior_knots = np.array([])
    
    # Create clamped knot vector
    knots = np.concatenate([
        np.zeros(degree + 1),  # p+1 repeated at start
        interior_knots,        # interior knots
        np.ones(degree + 1)    # p+1 repeated at end
    ])
    
    return torch.tensor(knots, dtype=torch.float32, device=device)


# Control points 
# control_points = torch.tensor([
#     [0.0, 0.0, 0.0],
#     [0.5, 1.5, 0.2],
#     [1.0, 0.0, -0.1],
#     [1.5, -1.5, 0.3],
#     [2.0, 0.0, 0.0],
#     [2.5, 1.5, -0.2],
#     [3.0, 0.0, 0.1],
#     [3.5, -1.5, 0.0],
#     [4.0, 0.0, -0.3],
#     [4.5, 1.0, 0.2]
# ], dtype=torch.float32)

# control_points = load_control_points()


# Load mesh
# mesh = Mesh("sphere.stl")
# N = 100
# fraction = N / mesh.npoints
# sampled = mesh.clone().subsample(fraction)
# if sampled.npoints > N:
#     sampled.points = sampled.points[:N]
# control_points_np = sampled.points
# control_points = torch.tensor(control_points_np, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Extracted {len(control_points)} surface points.")  

# Load mesh
# mesh = Mesh("sphere.stl")

# N = 430.5
# fraction = min(1.0, N / mesh.npoints)  # Make sure it's <= 1

# # Subsample points evenly from the mesh
# sampled = mesh.clone().subsample(fraction)

# if sampled.npoints > N:
#      sampled.points = sampled.points[:N]
# control_points_np = sampled.points
# control_points = torch.tensor(control_points_np, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
# # sorted_idx = torch.argsort(control_points[:, 0])
# # control_points = control_points[sorted_idx]
# print(f"Extracted {len(control_points)} surface points.")

# def reorder_nearest(points_np):
#     reordered = [points_np[0]]
#     points_np = np.delete(points_np, 0, axis=0)

#     while len(points_np) > 0:
#         dists = np.linalg.norm(points_np - reordered[-1], axis=1)
#         idx = np.argmin(dists)
#         reordered.append(points_np[idx])
#         points_np = np.delete(points_np, idx, axis=0)

#     return np.array(reordered)

# reordered_np = reorder_nearest(control_points_np)
# control_points = torch.tensor(reordered_np, dtype=torch.float32, device=control_points.device)





# Load and subsample mesh
mesh = Mesh("bunny.stl")
N = 500

# === Step 1: Sample N points uniformly from surface ===
# if N > mesh.npoints:
#     raise ValueError(f"Requested {N} points, but mesh only has {mesh.npoints} vertices.")

# indices = np.random.choice(mesh.npoints, N, replace=False)
# sampled = mesh.clone()
# sampled.points = sampled.points[indices]
indices = np.random.choice(mesh.npoints, N, replace=False)
sampled = mesh.clone().triangulate()
sampled.points = sampled.points[indices]
# === Step 2: Extract as NumPy array ===
control_points_np = sampled.points

# === Step 3: PCA sort along dominant direction ===
pca = PCA(n_components=1)
proj = pca.fit_transform(control_points_np)
sorted_idx = proj[:, 0].argsort()
control_points_np = control_points_np[sorted_idx]

# === Step 4 (Optional): Reorder using greedy nearest-neighbor ===
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

# === Step 5: Convert to PyTorch tensor ===
control_points = torch.tensor(
    control_points_np, dtype=torch.float32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Extracted and sorted {len(control_points)} control points.")

# === Step 6: Visualization ===
pointcloud = Points(control_points_np, r=8).c("red")
mesh.c("lightgray").alpha(0.3)

show(mesh, pointcloud, axes=1, title=" ")





degree = 3
# knot_vector = generate_knot_vector(len(control_points), degree)
knot_vector = generate_knot_vector(control_points, degree)

start3 = time.time()
t_values = torch.linspace(knot_vector[degree], knot_vector[-degree - 1], 600)
target_points = evaluate_bspline(control_points, knot_vector, degree, t_values)

print("Evaluate_bspline for target points took", time.time() - start3, "seconds")

# Optimisation 
start4 = time.time()
opt_cp = OptiBSpline(
    control_points=control_points,
    knots=knot_vector,
    degree=degree,
    # knot_vector=knot_vector,
    objective=lambda cp: bspline_objective(cp, target_points, degree, knot_vector),
    iters=100,
    lr=0.1
)
print("optimisation points took", time.time() - start4, "seconds")

start5 = time.time()
opt_cp2 = OptiBSpline(
    control_points=control_points,
    knots=knot_vector,
    degree=degree,
    # knot_vector=knot_vector,
    objective=lambda cp: bspline_objective2(cp, target_points, degree, knot_vector),
    iters=5,
    lr=0.1
)
print("optimisation points took", time.time() - start5, "seconds")

# print(torch.norm(opt_cp - opt_cp2))  

# Evaluate curves using the *optimized* control points


start = time.time()
opt_curve = evaluate_bspline(opt_cp, knot_vector, degree, t_values)
print("Evaluate_bspline1 took", time.time() - start, "seconds")
start2 = time.time()
opt_curve2 = evaluate_bspline(opt_cp2, knot_vector, degree, t_values)
print("Evaluate_bspline2 took", time.time() - start2, "seconds")


# Sample 20 points along curve
sample_indices = torch.linspace(0, len(opt_curve) - 1, steps=50).long()
# sampled_points = opt_curve2[sample_indices]


# Compute Gorkov potential 
# gorkov_vals = []
# for point in sampled_points:
#     p = point.view(1, 3).T.unsqueeze(0).to(device).to(DTYPE)  
#     print(p.shape)
#     wgs_out = wgs(p)
#     act = add_lev_sig(wgs_out)
#     target_pressure = 1000
#     act = act * (target_pressure / (act.abs().max() + 1e-10)) 
#     g_val = gorkov_analytical(act, p).item() 
#     gorkov_vals.append(g_val)
# gorkov_vals = torch.tensor(gorkov_vals)
# print("Gorkov values:", gorkov_vals)
# print("Min:", gorkov_vals.min().item(), "Max:", gorkov_vals.max().item())

# plot 1
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x_s = sampled_points[:, 0].detach().cpu().numpy()
# y_s = sampled_points[:, 1].detach().cpu().numpy()
# z_s = sampled_points[:, 2].detach().cpu().numpy()
# g_vals = gorkov_vals.detach().cpu().numpy()

# x_curve = opt_curve2[:, 0].detach().cpu().numpy()
# y_curve = opt_curve2[:, 1].detach().cpu().numpy()
# z_curve = opt_curve2[:, 2].detach().cpu().numpy()

# x_cp = control_points[:, 0].detach().cpu().numpy()
# y_cp = control_points[:, 1].detach().cpu().numpy()
# z_cp = control_points[:, 2].detach().cpu().numpy()

# # Gorkov potentials on sampled points
# scat = ax.scatter(x_s, y_s, z_s, c=g_vals, cmap='viridis',
#                   vmin=g_vals[0:49].min(), vmax=g_vals[0:49].max())
# plt.colorbar(scat, label='Gorkov Potential')

# # Full trajectory line
# ax.plot(x_curve, y_curve, z_curve, color='gray', linestyle='--', label='Optimised Path')

# # Original CPs
# ax.scatter(x_cp, y_cp, z_cp, color='red', label='Original CPs')

# ax.legend()
# ax.set_title("Gorkov Potential Along Particle Trajectory")
# plt.show()


#  plot 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot control points (red dots)
ax.scatter(control_points[:, 0].cpu(), 
           control_points[:, 1].cpu(), 
           control_points[:, 2].cpu(), 
           color='red', label='Control Points')


ax.plot(opt_curve[:, 0].cpu(), 
        opt_curve[:, 1].cpu(), 
        opt_curve[:, 2].cpu(), 
        color='blue', label='opt1', linewidth=1)

ax.plot(opt_curve2[:, 0].cpu(), 
        opt_curve2[:, 1].cpu(), 
        opt_curve2[:, 2].cpu(), 
        color='orange', label='opt2', linewidth=0.5)


ax.legend()
ax.set_title("Optimized Path and Control Points")
plt.show()


# single_point = opt_curve2[10]  # shape (3,)
# # Example: center the plane on your point's (x,z)
# center_x, center_y, _ = single_point.cpu().numpy()

# # Create plane corners shifted by center_x and center_z
# A, B, C = ABC(0.09, plane='xy')

# # Shift points by center
# def shift_points(points, dx, dy):
#     points_shifted = points.clone()
#     points_shifted[0] += dx  # x coordinate
#     points_shifted[1] += dy  # z coordinate
#     return points_shifted

# A = shift_points(A, center_x, center_y)
# B = shift_points(B, center_x, center_y)
# C = shift_points(C, center_x, center_y)
# positions = get_image_positions(A, B, C, res=(300, 300)) 
# # Reshape to [1,3,1] batch format
# point_tensor = single_point.view(1, 3, 1).to(DTYPE).to(device)

# # Compute activation for that point
# activation = wgs(point_tensor)
# activation = add_lev_sig(activation)

# # Get plane corners


# # Visualise
# Visualise(A, B, C, activation, points=point_tensor)

