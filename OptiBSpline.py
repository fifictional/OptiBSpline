# import torch
# from torch import Tensor
# from typing import List, Callable

# def bspline_basis(i: int, k: int, t: torch.Tensor, knots: torch.Tensor):
#     """
#     Recursively compute the B-spline basis function N_{i,k}(t).
    
#     Parameters:
#     - i: basis function index
#     - k: degree of the B-spline
#     - t: scalar position along the curve (tensor scalar)
#     - knots: knot vector (Tensor)
    
#     Returns:
#     - Scalar tensor value of the basis function at t
#     """
#     if k == 0:
#         left = knots[i]
#         right = knots[i + 1]
#         cond1 = (left <= t) & (t < right)
#         # For the last knot span, include right endpoint
#         cond2 = (i == len(knots) - 2) & (t == right)
#         return torch.where(cond1 | cond2, torch.tensor(1., device=t.device), torch.tensor(0., device=t.device))

#     else:
#         if i + k + 1 >= len(knots) or i + 1 >= len(knots):
#             return torch.tensor(0., device=t.device)
#         denom1 = knots[i+k] - knots[i]
#         denom2 = knots[i+k+1] - knots[i+1]

#         term1 = torch.where(
#             denom1 == 0,
#             torch.tensor(0., device=t.device),
#             ((t - knots[i]) / denom1) * bspline_basis(i, k-1, t, knots)
#         )
#         term2 = torch.where(
#             denom2 == 0,
#             torch.tensor(0., device=t.device),
#             ((knots[i+k+1] - t) / denom2) * bspline_basis(i+1, k-1, t, knots)
#         )
#         return term1 + term2



# def evaluate_bspline(control_points, knots, degree, t_values):
#     n_ctrl_pts = control_points.shape[0]
#     curve_points = []
#     for t in t_values:
#         point = torch.zeros(control_points.shape[1], dtype=control_points.dtype, device=control_points.device)
#         for i in range(n_ctrl_pts):
#             b = bspline_basis(i, degree, t, knots)  # no .item(), keep tensor scalar!
#             point += b * control_points[i]
#         curve_points.append(point)
#     return torch.stack(curve_points)



# def OptiBSpline(control_points: Tensor, knots: List[float], degree: int, knot_vector,
#                 objective: Callable, n: int = 100, iters: int = 200,
#                 optimiser=torch.optim.Adam, lr: float = 0.01,
#                 objective_params: dict = {}, log: bool = True) -> Tensor:
#     """
#     Optimise B-spline control points using given objective
#     """
#     ctrl = control_points.clone().detach().requires_grad_(True)
#     opt = optimiser([ctrl], lr=lr)
    
#     t_min = knot_vector[degree].item()
#     t_max = knot_vector[-degree - 1].item()
#     t_values = torch.linspace(t_min, t_max, n, device=control_points.device)

#     for i in range(iters):
#         opt.zero_grad()
        
#         with torch.set_grad_enabled(True):
#             curve_points = evaluate_bspline(ctrl, knots, degree, t_values)
#             loss = objective(curve_points, **objective_params)
        
    
#     for i in range(iters):
#         opt.zero_grad()
        
#         with torch.set_grad_enabled(True):
#             curve_points = evaluate_bspline(ctrl, knots, degree, t_values)
#             loss = objective(curve_points, **objective_params)
        
    
#         loss_dict = objective(curve_points, **objective_params)  # returns dict
#         loss = loss_dict["total"]
#         loss = torch.real(loss)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_([ctrl], max_norm=1.0)
        
#         opt.step()


#     return ctrl.detach()



import torch
from torch import Tensor
from typing import List, Callable

def bspline_basis(i: int, k: int, t: float, knots):
    """
    Recursively compute the B-spline basis function N_{i,k}(t).
    
    Parameters:
    - i: basis function index
    - k: degree of the B-spline
    - t: scalar position along the curve (float)
    - knots: knot vector (Tensor or list-like)

    Returns:
    - Scalar value of the basis function at t
    """
    t = float(t)
    if k == 0:
        left = knots[i].item()
        right = knots[i + 1].item()
        if left <= t < right:
            return 1.
        if t == knots[-1] and right == knots[-1]:
            return 1.0
        return 0.0

    else:
        if i + k + 1 >= len(knots) or i + 1 >= len(knots):
             return 0.0
        denom1 = knots[i+k] - knots[i]
        denom2 = knots[i+k+1] - knots[i+1]

        term1 = torch.where(
            denom1 == 0,
            0.0,
            (t - knots[i]) / denom1 * bspline_basis(i, k-1, t, knots)
        )
        term2 = torch.where(
            denom2 == 0,
            0.0,
            (knots[i+k+1] - t) / denom2 * bspline_basis(i+1, k-1, t, knots)
        )
        result = term1 + term2
        return result.real if isinstance(result, complex) else result



def evaluate_bspline(control_points, knots, degree, t_values):
    n_ctrl_pts = control_points.shape[0]
    curve_points = []
    for t in t_values:
        point = torch.zeros(control_points.shape[1], dtype=control_points.dtype, device=control_points.device)
        for i in range(n_ctrl_pts):
            b = bspline_basis(i, degree, t.item(), knots)
            # Ensure basis value is real and convert explicitly
            b_real = b.real if isinstance(b, complex) else b
            point += b_real * control_points[i]
        curve_points.append(point)
    return torch.stack(curve_points)


def OptiBSpline(control_points: Tensor, knots: List[float], degree: int, knot_vector,
                objective: Callable, n: int = 100, iters: int = 200,
                optimiser=torch.optim.Adam, lr: float = 0.01,
                objective_params: dict = {}, log: bool = True) -> Tensor:
    """
    Optimise B-spline control points using given objective
    """
    ctrl = control_points.clone().detach().requires_grad_(True)
    opt = optimiser([ctrl], lr=lr)
    
    t_min = knot_vector[degree].item()
    t_max = knot_vector[-degree - 1].item()
    t_values = torch.linspace(t_min, t_max, n, device=control_points.device)

    for i in range(iters):
        opt.zero_grad()
        
        with torch.set_grad_enabled(True):
            curve_points = evaluate_bspline(ctrl, knots, degree, t_values)
            loss_dict = objective(curve_points, **objective_params)
        
    
        # loss_dict = objective(curve_points, **objective_params)
        loss = loss_dict["total"]
        loss = torch.real(loss)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_([ctrl], max_norm=1.0)
        
        opt.step()

    return ctrl.detach()

import torch
from torch import Tensor
from typing import List, Callable

def bspline_basis_vectorized(degree: int, knots: Tensor, t_values: Tensor) -> Tensor:
    """
    Vectorized Cox-de Boor recursion to compute B-spline basis functions.

    Args:
        degree: spline degree (k)
        knots: knot vector, shape (m,)
        t_values: parameter values to evaluate, shape (n,)

    Returns:
        basis: shape (n, num_ctrl_pts)
    """
    device = knots.device
    n = t_values.shape[0]
    m = knots.shape[0]
    num_ctrl_pts = m - degree - 1

    # Initialize zeroth-degree basis functions N_{i,0}
    N = torch.zeros((n, m - 1), device=device)
    for i in range(m - 1):
        left = knots[i].item()
        right = knots[i + 1].item()
        cond = (t_values >= left) & (t_values < right)
        # Last knot span includes right endpoint
        if i == m - 2:
            cond = cond | (t_values == right)
        N[:, i] = cond.float()

    # Recursive computation for k=1..degree
    for k in range(1, degree + 1):
        N_new = torch.zeros((n, m - k - 1), device=device)
        for i in range(m - k - 1):
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            term1 = torch.zeros(n, device=device)
            term2 = torch.zeros(n, device=device)

            if denom1 > 0:
                term1 = ((t_values - knots[i]) / denom1) * N[:, i]
            if denom2 > 0:
                term2 = ((knots[i + k + 1] - t_values) / denom2) * N[:, i + 1]

            N_new[:, i] = term1 + term2
        N = N_new
        N = N.real

    return N  # shape (n, num_ctrl_pts)

def evaluate_bspline(control_points: Tensor, knots: Tensor, degree: int, t_values: Tensor) -> Tensor:
    """
    Evaluate B-spline curve at multiple t_values, vectorized.

    Args:
        control_points: shape (num_ctrl_pts, dim)
        knots: knot vector, shape (m,)
        degree: spline degree
        t_values: parameter values, shape (n,)

    Returns:
        curve_points: shape (n, dim)
    """
    N = bspline_basis_vectorized(degree, knots, t_values)  # (n, num_ctrl_pts)
    N = N.real.to(control_points.dtype)
    control_points = control_points.to(N.dtype)     
    return N @ control_points  # matrix multiply: (n, num_ctrl_pts) x (num_ctrl_pts, dim) = (n, dim)

def OptiBSpline(control_points: Tensor, knots: Tensor, degree: int,
                objective: Callable, n: int = 500, iters: int = 200,
                optimiser=torch.optim.Adam, lr: float = 0.01,
                objective_params: dict = {}, log: bool = True) -> Tensor:
    """
    Optimise B-spline control points using given objective.
    """
    ctrl = control_points.clone().detach().requires_grad_(True)
    opt = optimiser([ctrl], lr=lr)
    
    t_min = knots[degree].item()
    t_max = knots[-degree - 1].item()
    t_values = torch.linspace(t_min, t_max, n, device=control_points.device)

    for i in range(iters):
        opt.zero_grad()
        curve_points = evaluate_bspline(ctrl, knots, degree, t_values)
        loss_dict = objective(curve_points, **objective_params)
        loss = loss_dict["total"]
        loss = torch.real(loss) 

        loss.backward()
        print(f"the gradient norm: {ctrl.grad.norm()}")
        # torch.nn.utils.clip_grad_norm_([ctrl], max_norm=10000000.0)
        opt.step()
        if log and (i % max(1, iters // 10) == 0):
            print(f"Iter {i}: Loss {loss.item():.6f}")

        

    return ctrl.detach()



