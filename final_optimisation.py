import torch
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from OptiBSpline import evaluate_bspline, OptiBSpline
from acoustools.Utilities import TRANSDUCERS, device, DTYPE,  add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Gorkov import gorkov_analytical, get_gorkov_constants
from acoustools.Levitator import LevitatorController  
# from acoustools.Visualiser import Visualise, ABC, get_image_positions
from vedo import Mesh, show, Points
import numpy as np
from sklearn.decomposition import PCA
import numpy as np


def bspline_objective2(control_points, target_points, degree, knot_vector,
                     n_samples=600, alpha=10.0, beta=10.0, gamma=500000.0, delta=1.0):
    control_points = control_points.to(device).to(DTYPE)
    # control_points = control_points.clone().requires_grad_(True)
    
    t_values = torch.linspace(knot_vector[degree], knot_vector[-degree-1], n_samples,
                            device=control_points.device)
    curve_points = evaluate_bspline(control_points, knot_vector, degree, t_values)
    curve_points.retain_grad()
    
    # Smoothness loss - minimises acceleration + velocity
    with torch.no_grad():
        vel = curve_points[1:] - curve_points[:-1]
        total_length = vel.norm(dim=1).sum()
        target_length = total_length / vel.shape[0]
        target_length = (vel.norm(dim=1).sum() / vel.shape[0]).detach()

    loss_velocity = (vel.norm(dim=1) - target_length).square().sum()  
    accel = vel[1:] - vel[:-1]
    loss_accel = (accel**2).sum()
    loss_smooth = 0.1*loss_velocity + loss_accel  
    
    # Fit loss
    loss_fit = ((curve_points - target_points)**2).sum() if target_points is not None \
              else torch.tensor(0.0, device=device)
   
   # Gorkov loss
    gorkov_vals = []
    for point in curve_points:
        p = point.view(1, 3).T.unsqueeze(0).to(device).to(DTYPE)  
        wgs_out = wgs(p) 
        act = add_lev_sig(wgs_out) 
        act = act / (act.abs().max() + 1e-10)  
        # act = act / act.abs().max()
        g_val = gorkov_analytical(act, p)  
        gorkov_vals.append(g_val)

    gorkov_vals = torch.stack(gorkov_vals) 
    loss_gorkov = gorkov_vals.sum()  
    loss_gorkov = torch.abs(gorkov_vals).sum()  

    K1, K2 = get_gorkov_constants()
    print(f"K1: {K1}, K2: {K2}") 

    # Curvature loss -  minimises acceleration / velocity³
    vel = curve_points[1:] - curve_points[:-1]
    accel = vel[1:] - vel[:-1]
    curvature = (accel.norm(dim=1)**2) / (vel[:-1].norm(dim=1)**3 + 1e-6)
    loss_curvature = curvature.sum()

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

    with torch.no_grad():
        vel = curve_points[1:] - curve_points[:-1]
        total_length = vel.norm(dim=1).sum()
        target_length = total_length / vel.shape[0]
        target_length = (vel.norm(dim=1).sum() / vel.shape[0]).detach()

    vel = curve_points[1:] - curve_points[:-1]
    loss_velocity = (vel.norm(dim=1) - target_length).square().sum() 
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


def generate_knot_vector(control_points, degree):
    """Generate exact-length knot vector for given control points"""
    n = len(control_points)
    
    if isinstance(control_points, torch.Tensor):
        pts = control_points.cpu().numpy()
    else:
        pts = control_points
    
    if n > degree + 1:
        diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cum_length = np.concatenate([[0], np.cumsum(diffs)])
        interior = cum_length[1:-1] / cum_length[-1]  
        

        num_interior = n - degree - 1
        if num_interior > 0:
            interior_knots = np.linspace(0, 1, num_interior + 2)[1:-1] 
        else:
            interior_knots = np.array([])
    else:
        interior_knots = np.array([])
    

    knots = np.concatenate([
        np.zeros(degree + 1), 
        interior_knots,        
        np.ones(degree + 1)    
    ])
    
    return torch.tensor(knots, dtype=torch.float32, device=device)



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

control_points = torch.tensor(
    control_points_np, dtype=torch.float32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Extracted and sorted {len(control_points)} control points.")

pointcloud = Points(control_points_np, r=8).c("red")
mesh.c("lightgray").alpha(0.3)
show(mesh, pointcloud, axes=1, title=" ")


degree = 3
knot_vector = generate_knot_vector(control_points, degree)
t_values = torch.linspace(knot_vector[degree], knot_vector[-degree - 1], 600)
target_points = evaluate_bspline(control_points, knot_vector, degree, t_values)

# Optimisation 
opt_cp = OptiBSpline(
    control_points=control_points,
    knots=knot_vector,
    degree=degree,
    objective=lambda cp: bspline_objective(cp, target_points, degree, knot_vector),
    iters=100,
    lr=0.1
)

# opt_cp2 = OptiBSpline(
#     control_points=control_points,
#     knots=knot_vector,
#     degree=degree,
#     objective=lambda cp: bspline_objective2(cp, target_points, degree, knot_vector),
#     iters=5,
#     lr=0.1
# )


opt_curve = evaluate_bspline(opt_cp, knot_vector, degree, t_values)
# opt_curve2 = evaluate_bspline(opt_cp2, knot_vector, degree, t_values)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(control_points[:, 0].cpu(), 
           control_points[:, 1].cpu(), 
           control_points[:, 2].cpu(), 
           color='red', label='Control Points')


ax.plot(opt_curve[:, 0].cpu(), 
        opt_curve[:, 1].cpu(), 
        opt_curve[:, 2].cpu(), 
        color='blue', label='opt1', linewidth=1)

# ax.plot(opt_curve2[:, 0].cpu(), 
#         opt_curve2[:, 1].cpu(), 
#         opt_curve2[:, 2].cpu(), 
#         color='orange', label='opt2', linewidth=0.5)

ax.legend()
ax.set_title("Optimised Path and Control Points")
plt.show()



lev = LevitatorController(ids=(53, 73))  
print("connected!")
lev.set_frame_rate(1000)

p = opt_curve.T.unsqueeze(0).to(device).to(DTYPE)  
wgs_out = wgs(p)
activation = add_lev_sig(wgs_out)
activation = activation / (activation.abs().max() + 1e-10)
activation_np = activation.detach().cpu().numpy()
lev.levitate(activation_np, num_loops=5)


