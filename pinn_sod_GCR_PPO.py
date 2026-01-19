"""
Priority-Aware PCGrad for PINN: Sod's Shock Tube Problem
--------------------------------------------------------
Original Author: Mohammad E. Heravifard
Modified by: AI Assistant
Algorithm: Priority-Aware Gradient Conflict Resolution (inspired by GCR-PPO)

Description:
This script solves the 1D Euler equations for the Sod Shock Tube problem using a 
Physics-Informed Neural Network (PINN). It uses a custom optimizer (PriorityPCGrad) 
to resolve gradient conflicts between the Initial Conditions (High Priority) and 
the PDE residuals (Low Priority).
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from scipy.optimize import brentq
import random
import copy

# ------------------------------------------------------------------------------------------------------------------------------------#
# CONFIGURATION
# ------------------------------------------------------------------------------------------------------------------------------------#

# Set random seeds for reproducibility
SEED = 123456
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -------------------------------------------------------------------------------------------------------------------------------------#
# OPTIMIZER: PRIORITY-AWARE PCGrad
# -------------------------------------------------------------------------------------------------------------------------------------#

class PriorityPCGrad:
    """
    Priority-Aware PCGrad optimizer wrapper.
    
    Logic:
    1. Compute gradients for all tasks independenty.
    2. Check for conflicts (negative cosine similarity).
    3. If Task A conflicts with Task B:
       - If Priority(B) > Priority(A): Project A's gradient onto the normal plane of B 
         (Remove the component of A that harms B).
       - If Priority(B) == Priority(A): Project both (Standard PCGrad behavior).
       - If Priority(B) < Priority(A): Do nothing to A (A dominates).
    """
    def __init__(self, optimizer):
        self._optim = optimizer

    def zero_grad(self):
        return self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives, priorities):
        """
        Args:
            objectives (list): List of loss tensors (scalars) for each task.
            priorities (list): List of integers indicating priority level. Higher is better.
        """
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads, priorities)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, priorities):
        final_grads = []
        
        # Determine the order of comparison. We shuffle to ensure fairness among equal-priority tasks.
        indices = list(range(len(grads)))
        random.shuffle(indices)

        for i in range(len(grads)):
            g_i = grads[i].clone()
            p_i = priorities[i]
            
            for j in indices:
                if i == j: continue
                
                g_j = grads[j]
                p_j = priorities[j]
                
                # Dot product determines if gradients are conflicting (< 0)
                g_i_g_j = torch.dot(g_i, g_j)
                
                if g_i_g_j < 0:
                    if p_j > p_i:
                        # Case 1: j is higher priority. i must yield.
                        # Project g_i away from g_j's gradient direction
                        g_i -= (g_i_g_j) * g_j / (g_j.norm()**2 + 1e-8)
                    
                    elif p_j == p_i:
                        # Case 2: Equal priority. Mutual compromise (Standard PCGrad).
                        g_i -= (g_i_g_j) * g_j / (g_j.norm()**2 + 1e-8)
                    
                    # Case 3: p_j < p_i. Do nothing. i is dominant.

            final_grads.append(g_i)
        
        # Sum up all projected gradients for the final update
        merged_grad = torch.zeros_like(final_grads[0])
        for g in final_grads:
            merged_grad += g
            
        return merged_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for loss in objectives:
            self._optim.zero_grad()
            loss.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

# -------------------------------------------------------------------------------------------------------------------------------------#
# NEURAL NETWORK MODEL
# -------------------------------------------------------------------------------------------------------------------------------------#

class DNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=3, num_hidden_layers=6):
        super(DNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def compute_pde_losses(self, x):
        """
        Returns individual PDE residuals [Mass, Momentum, Energy]
        """
        y = self.net(x)
        rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        gamma = 1.4

        drho_dt_dx = compute_gradients(rho, x)
        rho_t, rho_x = drho_dt_dx[:, :1], drho_dt_dx[:, 1:]

        du_dt_dx = compute_gradients(u, x)
        u_t, u_x = du_dt_dx[:, :1], du_dt_dx[:, 1:]

        dp_dt_dx = compute_gradients(p, x)
        p_t, p_x = dp_dt_dx[:, :1], dp_dt_dx[:, 1:]

        mass_residual = rho_t + u * rho_x + rho * u_x
        momentum_residual = rho * (u_t + u * u_x) + p_x
        energy_residual = p_t + gamma * p * u_x + u * p_x

        return [
            (mass_residual**2).mean(),
            (momentum_residual**2).mean(),
            (energy_residual**2).mean()
        ]

    def compute_ic_losses(self, x_ic, rho_ic, u_ic, p_ic):
        """
        Returns individual Initial Condition errors [Rho_err, P_err, U_err]
        """
        y_ic = self.net(x_ic)
        rho_ic_nn, p_ic_nn, u_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2]

        return [
            ((rho_ic_nn - rho_ic) ** 2).mean(),
            ((p_ic_nn - p_ic) ** 2).mean(),
            ((u_ic_nn - u_ic) ** 2).mean()
        ]

# -----------------------------------------------------------------------------------------------------------------------------------#
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------------------------#

def compute_gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True)[0]

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f'Expected torch.Tensor or np.ndarray, got {type(tensor)}')

def initial_conditions(x):
    N = x.shape[0]
    rho_init = np.zeros(N)
    u_init = np.zeros(N)
    p_init = np.zeros(N)
    for i in range(N):
        if x[i] <= 0.5:
            rho_init[i] = 1.0
            p_init[i] = 1.0
        else:
            rho_init[i] = 0.125
            p_init[i] = 0.1
    return rho_init, u_init, p_init

def get_exact_sod(x_np, t_val, gamma=1.4):
    """Analytical Solution for Sod Shock Tube"""
    rho_L, p_L, u_L = 1.0, 1.0, 0.0
    rho_R, p_R, u_R = 0.125, 0.1, 0.0
    x_0 = 0.5

    if t_val <= 1e-9:
        rho = np.where(x_np <= x_0, rho_L, rho_R)
        u = np.where(x_np <= x_0, u_L, u_R)
        p = np.where(x_np <= x_0, p_L, p_R)
        return rho, p, u

    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    def f(P, p_k, rho_k, a_k):
        if P <= p_k: return 2*a_k/(gamma-1) * ((P/p_k)**((gamma-1)/(2*gamma)) - 1)
        else:
            ak = 2/((gamma+1)*rho_k)
            bk = (gamma-1)/(gamma+1)*p_k
            return (P - p_k) * np.sqrt(ak/(P + bk))

    p_star = brentq(lambda P: f(P, p_L, rho_L, a_L) + f(P, p_R, rho_R, a_R) + u_R - u_L, 1e-6, 5.0)
    u_star = 0.5*(u_L + u_R) + 0.5*(f(p_star, p_R, rho_R, a_R) - f(p_star, p_L, rho_L, a_L))

    rho_star_L = rho_L * (p_star/p_L)**(1/gamma)
    rho_star_R = rho_R * ((p_star/p_R + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1)*p_star/p_R + 1))
    s_shock = u_R + a_R * np.sqrt((gamma+1)/(2*gamma)*(p_star/p_R) + (gamma-1)/(2*gamma))
    head_fan = u_L - a_L
    tail_fan = u_star - a_L * (p_star/p_L)**((gamma-1)/(2*gamma))

    rho_sol, p_sol, u_sol = np.zeros_like(x_np), np.zeros_like(x_np), np.zeros_like(x_np)
    xi = (x_np - x_0) / t_val

    for i, x_val in enumerate(xi):
        if x_val < head_fan: rho_sol[i], p_sol[i], u_sol[i] = rho_L, p_L, u_L
        elif x_val < tail_fan:
            u_sol[i] = 2/(gamma+1) * (a_L + (gamma-1)/2*u_L + x_val)
            a_fan = u_sol[i] - x_val
            rho_sol[i] = rho_L * (a_fan/a_L)**(2/(gamma-1))
            p_sol[i] = p_L * (a_fan/a_L)**((2*gamma)/(gamma-1))
        elif x_val < u_star: rho_sol[i], p_sol[i], u_sol[i] = rho_star_L, p_star, u_star
        elif x_val < s_shock: rho_sol[i], p_sol[i], u_sol[i] = rho_star_R, p_star, u_star
        else: rho_sol[i], p_sol[i], u_sol[i] = rho_R, p_R, u_R
    return rho_sol, p_sol, u_sol

def evaluate_model(model, x_test, num_x, num_t):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
    pred_np = to_numpy(predictions)
    rho_pred = pred_np[:, 0].reshape(num_x, num_t)
    p_pred = pred_np[:, 1].reshape(num_x, num_t)
    u_pred = pred_np[:, 2].reshape(num_x, num_t)
    return rho_pred, p_pred, u_pred

def plot_comparison_with_error(t, x, rho_pred, p_pred, u_pred, t_eval=0.2):
    t_idx = np.argmin(np.abs(t - t_eval))
    rho_pinn = rho_pred[:, t_idx]
    p_pinn = p_pred[:, t_idx]
    u_pinn = u_pred[:, t_idx]
    rho_ex, p_ex, u_ex = get_exact_sod(x, t_eval)

    def calc_l2(pred, exact):
        return np.sqrt(np.mean((pred - exact)**2)) / np.sqrt(np.mean(exact**2))

    err_rho = calc_l2(rho_pinn, rho_ex)
    err_p = calc_l2(p_pinn, p_ex)
    err_u = calc_l2(u_pinn, u_ex)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'PriorityPCGrad Solution at t={t_eval}', fontsize=16)
    
    titles = ['Density', 'Pressure', 'Velocity']
    pinn_data = [rho_pinn, p_pinn, u_pinn]
    exact_data = [rho_ex, p_ex, u_ex]
    errors = [err_rho, err_p, err_u]
    colors = ['blue', 'red', 'green']

    for i in range(3):
        axs[0, i].plot(x, exact_data[i], 'k--', label='Exact', linewidth=2)
        axs[0, i].plot(x, pinn_data[i], color=colors[i], label='PINN', alpha=0.8, linewidth=2)
        axs[0, i].set_title(f'{titles[i]} (Rel L2: {errors[i]:.2e})')
        axs[0, i].legend()
        axs[0, i].grid(True, alpha=0.3)

        abs_err = np.abs(pinn_data[i] - exact_data[i])
        axs[1, i].plot(x, abs_err, color=colors[i], linewidth=1.5)
        axs[1, i].fill_between(x, abs_err, color=colors[i], alpha=0.2)
        axs[1, i].set_title(f'{titles[i]} Abs Error')
        axs[1, i].set_yscale('log')
        axs[1, i].grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------#
# TRAINING FUNCTION
# -----------------------------------------------------------------------------------------------------------------------------------#

def train_model(model, optimizer, scheduler, x_int_train, x_ic_train, rho_ic, u_ic, p_ic, epochs):
    
    # WRAP OPTIMIZER WITH PRIORITY PCGRAD
    pc_optimizer = PriorityPCGrad(optimizer)

    # DEFINE PRIORITIES
    # PDE Losses (Indices 0, 1, 2) -> Priority 1 (Low)
    # IC Losses  (Indices 3, 4, 5) -> Priority 2 (High)
    # Explanation: We want to satisfy Initial Conditions FIRST. If PDE gradients 
    # conflict with IC gradients, PDE gradients will be projected (modified).
    task_priorities = [1, 1, 1, 2, 2, 2]

    for epoch in range(1, epochs + 1):
        model.train()
        
        # 1. Compute Loss LISTS (Do not sum them yet)
        pde_losses = model.compute_pde_losses(x_int_train)           # [L_mass, L_mom, L_energy]
        ic_losses = model.compute_ic_losses(x_ic_train, rho_ic, u_ic, p_ic) # [L_rho, L_p, L_u]
        
        # 2. Combine for optimizer
        # We can still apply scalar weights here if we want to change magnitude, 
        # but PCGrad handles direction. Let's weigh ICs slightly higher in magnitude too.
        weighted_ic_losses = [l * 5.0 for l in ic_losses]
        all_objectives = pde_losses + weighted_ic_losses
        
        # 3. Backward Pass with Priorities
        pc_optimizer.pc_backward(all_objectives, task_priorities)
        
        # 4. Update
        pc_optimizer.step()
        scheduler.step()

        # Monitoring
        if epoch % 500 == 0 or epoch == 1:
            loss_pde_val = sum([l.item() for l in pde_losses])
            loss_ic_val = sum([l.item() for l in ic_losses])
            print(f"Epoch {epoch:05d}: PDE Loss: {loss_pde_val:.6f} | IC Loss: {loss_ic_val:.6f}")

# -----------------------------------------------------------------------------------------------------------------------------------#
# MAIN
# -----------------------------------------------------------------------------------------------------------------------------------#

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Hyperparameters
    learning_rate = 0.001
    epochs = 5000  
    num_x = 1000
    num_t = 1000
    num_ic_samples = 2000
    num_f_samples = 15000

    # Grid Generation
    x = np.linspace(0, 1, num_x)
    t = np.linspace(0, 0.2, num_t)
    t_grid, x_grid = np.meshgrid(t, x, indexing='ij') # 显式指定 indexing='ij' 避免歧义
    
    T = t_grid.flatten()[:, None] # Shape: (1000000, 1)
    X = x_grid.flatten()[:, None] # Shape: (1000000, 1)

    # 2. Training Data - Initial Conditions (IC)
    # t=0, x 随机采样
    ic_indices = np.random.choice(num_x, num_ic_samples, replace=True)
    x_ic = x[ic_indices][:, None]      # 从 x 向量直接取更安全
    t_ic = np.zeros_like(x_ic)         # t 全为 0
    x_ic_train = np.hstack((t_ic, x_ic)) # Shape: (2000, 2)
    
    # 计算 IC 的真实值
    rho_ic_train_np, u_ic_train_np, p_ic_train_np = initial_conditions(x_ic.flatten())

    # 3. Training Data - PDE (Collocation Points)
    # 在时空内部随机采样
    total_points = num_x * num_t
    f_indices = np.random.choice(total_points, num_f_samples, replace=False)
    
    t_int = T[f_indices] # Shape: (15000, 1)
    x_int = X[f_indices] # Shape: (15000, 1)
    
    # !!! 关键修正：确保是水平拼接 (N, 2) !!!
    x_int_train = np.hstack((t_int, x_int))
    # === DEBUG PRINT ===
    print(f"Shape check:")
    print(f"  t_int: {t_int.shape}")
    print(f"  x_int: {x_int.shape}")
    print(f"  x_int_train (Expected 15000, 2): {x_int_train.shape}")
    if x_int_train.shape[1] != 2:
        raise ValueError("x_int_train dimensions are wrong! Must be (N, 2).")
    # ===================

    # Test Data (用于最终画图)
    # 生成规则网格用于评估
    x_test_grid, t_test_grid = np.meshgrid(x, t, indexing='ij') # 注意顺序保持一致
    X_test = x_test_grid.flatten()[:, None]
    T_test = t_test_grid.flatten()[:, None]
    x_test = np.hstack((T_test, X_test))

    # Convert to Tensor
    x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32, device=device)
    # 启用梯度计算，因为 PDE 需要求导
    x_int_train = torch.tensor(x_int_train, dtype=torch.float32, requires_grad=True, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    
    rho_ic_train = torch.tensor(rho_ic_train_np, dtype=torch.float32, device=device)
    u_ic_train = torch.tensor(u_ic_train_np, dtype=torch.float32, device=device)
    p_ic_train = torch.tensor(p_ic_train_np, dtype=torch.float32, device=device)

    # Model Init
    model = DNN(hidden_dim=50, num_hidden_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.8)

    # Train
    print("Starting training with Priority-Aware PCGrad...")
    start_time = time.time()
    train_model(model, optimizer, scheduler, x_int_train, x_ic_train,
                rho_ic_train, u_ic_train, p_ic_train, epochs)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Evaluate
    print("Evaluating...")
    rho_pred, p_pred, u_pred = evaluate_model(model, x_test, num_x, num_t)
    plot_comparison_with_error(t, x, rho_pred, p_pred, u_pred, t_eval=0.2)

if __name__ == '__main__':
    main()