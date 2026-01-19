"""
Project: Data-Free Physics-Driven Neural Networks Approach for 1D Euler Equations
Method: Selective Task Group Updates (STGU) for PINN Optimization
Modified based on: [Jeong & Yoon, ICLR 2025]
"""
# -----------------------------------------------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from scipy.optimize import brentq
from sklearn.cluster import AffinityPropagation # 需要安装 scikit-learn
import random

# ------------------------------------------------------------------------------------------------------------------------------------#

# SET RANDOM SEEDS
torch.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)

# -------------------------------------------------------------------------------------------------------------------------------------#

# 1. DEFINING THE STGU COMPONENT (核心算法移植)
class TaskAffinity:
    """
    Manages task affinity matrix and dynamic grouping for Selective Task Group Updates.
    Ported and simplified from sel-update-mtl repository.
    """
    def __init__(self, tasks, affin_decay=0.9, preference=None):
        self.tasks = tasks
        self.affin_decay = affin_decay
        self.preference = preference
        self.affinity_map = torch.zeros(len(self.tasks), len(self.tasks))
        self.pre_loss = {task: -1.0 for task in tasks}
        self.group = {f'group1': tasks} # Initial group: all tasks together
        self.convergence_iter = 50

    def update(self, current_group_name, loss_dict):
        """Update affinity map based on how loss changed after an optimization step."""
        current_group_tasks = self.group[current_group_name]
        
        for task_s in current_group_tasks: # Source task (updated)
            for task_t in self.tasks:      # Target task (observed)
                if self.pre_loss[task_t] <= 0: continue
                
                # Calculate relative loss improvement (affinity)
                # affin > 0 means improvement, affin < 0 means degradation (conflict)
                current_val = loss_dict[task_t].item()
                prev_val = self.pre_loss[task_t]
                
                # Formula derived from the repo
                change_ratio = 1 - current_val / prev_val
                
                # Normalize by group size (approximate expectation)
                affin_raw = change_ratio / len(current_group_tasks)
                
                # Logic to handle self-loops and cross-task affinity
                if task_t in current_group_tasks:
                     # For tasks in the same group, we check consistency
                     # Detailed logic simplified from repo for stability
                    final_affin = affin_raw
                else:
                    final_affin = affin_raw

                # Update the affinity map with Exponential Moving Average
                idx_s = self.tasks.index(task_s)
                idx_t = self.tasks.index(task_t)
                self.affinity_map[idx_s, idx_t] = (1 - self.affin_decay) * self.affinity_map[idx_s, idx_t] + self.affin_decay * final_affin

        # Update previous loss for next step
        for task in self.tasks:
            self.pre_loss[task] = loss_dict[task].item()

    def init_pre_loss(self):
        """Reset previous loss tracker at the start of an iteration."""
        for task in self.tasks:
            self.pre_loss[task] = -1.0

    def next_group(self):
        """Perform Affinity Propagation to determine new groups."""
        convergence_iter = self.convergence_iter
        
        # 1. 获取 numpy 数组并确保可写
        X = np.array(self.affinity_map.detach().cpu().numpy())
        
        # 2. 处理对角线
        n = X.shape[0]
        for i in range(n):
            if X[i, i] == 0:
                X[i, i] = 1.0
        
        # 3. 归一化
        for i in range(n):
            if X[i, i] != 0:
                X[:, i] /= X[i, i]
        
        X = (X + X.T) / 2 # 对称化

        # [新增修复] 添加微小噪声以消除 "All samples have mutually equal similarities" 警告
        # 这有助于打破完全对称的状态，让聚类算法能正常收敛
        X += 1e-6 * np.random.randn(*X.shape)
        
        # 4. 动态聚类
        res = {}
        for _ in range(5):
            try:
                cluster = AffinityPropagation(preference=self.preference, affinity='precomputed', 
                                              convergence_iter=convergence_iter, random_state=None)
                cluster.fit(X)
                labels = cluster.labels_
                cluster_centers = cluster.cluster_centers_indices_
                
                if len(cluster_centers) > 0:
                    for i, center in enumerate(cluster_centers):
                        members = [self.tasks[j] for j, label in enumerate(labels) if label == i]
                        res[f'group{i+1}'] = members
                    break
            except Exception:
                pass
            convergence_iter += 50
        
        # 兜底策略：如果聚类失败，所有任务归为一组
        if len(res) == 0:
            res = {'group1': self.tasks}
            
        self.group = res
        train_group_names = list(self.group.keys())
        random.shuffle(train_group_names) 
        return train_group_names
# -------------------------------------------------------------------------------------------------------------------------------------#

# DEFINING THE NEURAL NETWORK MODEL
class DNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=30, output_dim=3, num_hidden_layers=5):
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

    def compute_all_losses(self, x_int, x_ic, rho_ic_true, u_ic_true, p_ic_true):
        """
        Compute all individual loss components for STGU.
        Returns a dictionary: {'mass': ..., 'momentum': ..., 'energy': ..., 'ic_rho': ..., ...}
        """
        loss_dict = {}
        gamma = 1.4

        # --- 1. PDE Losses (Interior) ---
        y = self.net(x_int)
        rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:]

        drho_dt_dx = compute_gradients(rho, x_int)
        rho_t, rho_x = drho_dt_dx[:, :1], drho_dt_dx[:, 1:]

        du_dt_dx = compute_gradients(u, x_int)
        u_t, u_x = du_dt_dx[:, :1], du_dt_dx[:, 1:]

        dp_dt_dx = compute_gradients(p, x_int)
        p_t, p_x = dp_dt_dx[:, :1], dp_dt_dx[:, 1:]

        mass_residual = rho_t + u * rho_x + rho * u_x
        momentum_residual = rho * (u_t + u * u_x) + p_x
        energy_residual = p_t + gamma * p * u_x + u * p_x

        loss_dict['mass'] = (mass_residual**2).mean()
        loss_dict['momentum'] = (momentum_residual**2).mean()
        loss_dict['energy'] = (energy_residual**2).mean()

        # --- 2. IC Losses (Boundary) ---
        y_ic = self.net(x_ic)
        rho_ic_nn, p_ic_nn, u_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2]

        loss_dict['ic_rho'] = ((rho_ic_nn - rho_ic_true) ** 2).mean()
        loss_dict['ic_u'] = ((u_ic_nn - u_ic_true) ** 2).mean()
        loss_dict['ic_p'] = ((p_ic_nn - p_ic_true) ** 2).mean()

        # Helper for total loss tracking
        loss_dict['total'] = sum(loss_dict.values())
        
        return loss_dict

# -----------------------------------------------------------------------------------------------------------------------------------#

# UTILITY FUNCTIONS
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

# -----------------------------------------------------------------------------------------------------------------------------------#

# MODIFIED TRAINING FUNCTION WITH STGU
def train_model_stgu(model, optimizer, scheduler, x_int_train, x_ic_train, rho_ic, u_ic, p_ic, epochs, device):
    """
    Train using Selective Task Group Updates.
    """
    # 1. Define Tasks
    tasks = ['mass', 'momentum', 'energy', 'ic_rho', 'ic_u', 'ic_p']
    
    # 2. Initialize Affinity Manager
    # affin_decay controls how fast the map updates (0.9 is standard in the paper)
    task_affinity = TaskAffinity(tasks, affin_decay=0.9) 

    print(f"Starting STGU Training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # --- STGU Step 1: Determine Groups ---
        # Before optimization, decide how to group tasks based on history
        # (For the very first steps, it uses random or single group)
        if epoch <= 10: 
             # Warmup with random groups or all together to populate affinity
            train_group_names = list(task_affinity.group.keys())
            random.shuffle(train_group_names)
        else:
            train_group_names = task_affinity.next_group()

        # Reset pre-loss for this iteration's calculation
        task_affinity.init_pre_loss()
        
        # --- STGU Step 2: Sequential Group Updates ---
        # The key idea: Update model parameters FOR EACH GROUP sequentially
        
        # Log total loss for printing
        total_loss_val = 0
        
        for group_name in train_group_names:
            current_tasks = task_affinity.group[group_name]
            
            # A. Forward Pass & Loss Calculation
            # We must do forward pass inside the loop because params change after each group update
            optimizer.zero_grad()
            loss_dict = model.compute_all_losses(x_int_train, x_ic_train, rho_ic, u_ic, p_ic)
            
            # B. Construct Group Loss
            group_loss = torch.stack([loss_dict[t] for t in current_tasks]).sum()
            
            # C. Backward & Update (ONLY for this group)
            group_loss.backward()
            optimizer.step()
            
            # D. Update Affinity Map (Observer Effect)
            # We check how this update affected ALL tasks (requires new forward pass or approx)
            # The official implementation uses the loss values from THIS step to update affinity for NEXT step.
            # Ideally, one would measure loss *after* update to see effect, but the repo implementation
            # updates the map using the current loss values relative to the previous step's tracker.
            # To strictly follow repo logic:
            task_affinity.update(group_name, loss_dict)
            
            # Accumulate for logging (just using the last computed values)
            if group_name == train_group_names[-1]:
                total_loss_val = loss_dict['total'].item()

        scheduler.step()

        if epoch % 100 == 0:
            groups_str = str({k: v for k, v in task_affinity.group.items()})
            print(f"Epoch {epoch:04d}: Total Loss = {total_loss_val:.6f} | Groups: {len(task_affinity.group)}")
            # Uncomment to see dynamic grouping details:
            # print(f"  Current Groups: {groups_str}")

# -----------------------------------------------------------------------------------------------------------------------------------#
# (Exact Solution and Plotting functions remain unchanged)
def get_exact_sod(x_np, t_val, gamma=1.4):
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
        if P <= p_k: 
            return 2*a_k/(gamma-1) * ((P/p_k)**((gamma-1)/(2*gamma)) - 1)
        else: 
            ak = 2/((gamma+1)*rho_k)
            bk = (gamma-1)/(gamma+1)*p_k
            return (P - p_k) * np.sqrt(ak/(P + bk))

    try:
        p_star = brentq(lambda P: f(P, p_L, rho_L, a_L) + f(P, p_R, rho_R, a_R) + u_R - u_L, 1e-6, 5.0)
    except:
        p_star = 0.5 # Fallback
        
    u_star = 0.5*(u_L + u_R) + 0.5*(f(p_star, p_R, rho_R, a_R) - f(p_star, p_L, rho_L, a_L))

    rho_star_L = rho_L * (p_star/p_L)**(1/gamma)
    rho_star_R = rho_R * ((p_star/p_R + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1)*p_star/p_R + 1))
    s_shock = u_R + a_R * np.sqrt((gamma+1)/(2*gamma)*(p_star/p_R) + (gamma-1)/(2*gamma))
    head_fan = u_L - a_L
    tail_fan = u_star - a_L * (p_star/p_L)**((gamma-1)/(2*gamma))

    rho_sol, p_sol, u_sol = np.zeros_like(x_np), np.zeros_like(x_np), np.zeros_like(x_np)
    xi = (x_np - x_0) / t_val

    for i, x_val in enumerate(xi):
        if x_val < head_fan: 
            rho_sol[i], p_sol[i], u_sol[i] = rho_L, p_L, u_L
        elif x_val < tail_fan: 
            u_sol[i] = 2/(gamma+1) * (a_L + (gamma-1)/2*u_L + x_val)
            a_fan = u_sol[i] - x_val
            rho_sol[i] = rho_L * (a_fan/a_L)**(2/(gamma-1))
            p_sol[i] = p_L * (a_fan/a_L)**((2*gamma)/(gamma-1))
        elif x_val < u_star: 
            rho_sol[i], p_sol[i], u_sol[i] = rho_star_L, p_star, u_star
        elif x_val < s_shock: 
            rho_sol[i], p_sol[i], u_sol[i] = rho_star_R, p_star, u_star
        else: 
            rho_sol[i], p_sol[i], u_sol[i] = rho_R, p_R, u_R
            
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
    titles = ['Density', 'Pressure', 'Velocity']
    pinn_data = [rho_pinn, p_pinn, u_pinn]
    exact_data = [rho_ex, p_ex, u_ex]
    errors = [err_rho, err_p, err_u]
    colors = ['blue', 'red', 'green']

    for i in range(3):
        axs[0, i].plot(x, exact_data[i], 'k--', label='Exact Solution', linewidth=2)
        axs[0, i].plot(x, pinn_data[i], color=colors[i], label='PINN (STGU)', alpha=0.7)
        axs[0, i].set_title(f'{titles[i]} at t={t_eval}\nRel. L2 Error: {errors[i]:.4e}')
        axs[0, i].legend()
        axs[0, i].grid(True, alpha=0.3)
        abs_err = np.abs(pinn_data[i] - exact_data[i])
        axs[1, i].fill_between(x, abs_err, color=colors[i], alpha=0.2)
        axs[1, i].plot(x, abs_err, color=colors[i], linewidth=1)
        axs[1, i].set_title(f'{titles[i]} Absolute Error')
        axs[1, i].set_yscale('log')
        axs[1, i].set_xlabel('Position (x)')
        axs[1, i].grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------#

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    learning_rate = 0.001       # STGU often handles higher LRs better, but sticking to conservative
    epochs = 5000               # Reduced slightly as STGU does more updates per epoch
    num_x = 1000
    num_t = 1000
    num_ic_samples = 1000
    num_f_samples = 11000

    x = np.linspace(0, 1, num_x)
    t = np.linspace(0, 0.2, num_t)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]

    ic_indices = np.random.choice(num_x, num_ic_samples, replace=False)
    x_ic = x_grid[ic_indices, 0][:, None]
    t_ic = t_grid[ic_indices, 0][:, None]
    x_ic_train = np.hstack((t_ic, x_ic))
    rho_ic_train_np, u_ic_train_np, p_ic_train_np = initial_conditions(x_ic.flatten())

    f_indices = np.random.choice(num_x * num_t, num_f_samples, replace=False)
    x_int = X[f_indices, None].reshape(-1, 1)
    t_int = T[f_indices, None].reshape(-1, 1)
    x_int_train = np.hstack((t_int, x_int))

    x_test = np.hstack((T, X))

    x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32, device=device)
    x_int_train = torch.tensor(x_int_train, dtype=torch.float32, requires_grad=True, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    rho_ic_train = torch.tensor(rho_ic_train_np, dtype=torch.float32, device=device)
    u_ic_train = torch.tensor(u_ic_train_np, dtype=torch.float32, device=device)
    p_ic_train = torch.tensor(p_ic_train_np, dtype=torch.float32, device=device)

    model = DNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

    start_time = time.time()
    
    # Use the new STGU training function
    train_model_stgu(model, optimizer, scheduler, x_int_train, x_ic_train,
                     rho_ic_train, u_ic_train, p_ic_train, epochs, device)
                     
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    print("Evaluating model on the full domain...")
    rho_pred, p_pred, u_pred = evaluate_model(model, x_test, num_x, num_t)

    plot_comparison_with_error(t, x, rho_pred, p_pred, u_pred, t_eval=0.2)

if __name__ == '__main__':
    main()