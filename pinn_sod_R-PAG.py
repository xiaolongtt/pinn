"""
Supervisor: Prof. K. Hejranfar
Author: Mohammad E. Heravifard (Modified with R-PAG Algorithm)
Project #01: Data-Free Physics-Driven Neural Networks Approach
for 1D Euler Equations for Compressible Flows:
Testcase #01: Sod's Shock Tube Problem.
Algorithm: R-PAG (Residual-driven Asymmetric Gradient Projection) v4.0
"""
# -----------------------------------------------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from scipy.optimize import brentq

# ------------------------------------------------------------------------------------------------------------------------------------#

# SET RANDOM SEEDS
torch.manual_seed(123456)
np.random.seed(123456)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------------------------------------------------------------------#

# UTILITY FUNCTIONS (Gradients)
def compute_gradients(outputs, inputs):
    """一阶导数"""
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True)[0]

def compute_second_derivative(first_grads, inputs):
    """二阶导数 (用于人工粘性项)"""
    # first_grads 已经是 [dt, dx]，我们需要对 x (索引1) 再求导
    grad_x = first_grads[:, 1:2]
    second_grads = torch.autograd.grad(grad_x, inputs,
                                       grad_outputs=torch.ones_like(grad_x),
                                       create_graph=True, retain_graph=True)[0]
    return second_grads[:, 1:2] # 返回 u_xx

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
            rho_init[i] = 1.0; p_init[i] = 1.0
        else:
            rho_init[i] = 0.125; p_init[i] = 0.1
    return rho_init, u_init, p_init

# -------------------------------------------------------------------------------------------------------------------------------------#

# DEFINING THE NEURAL NETWORK MODEL
class DNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=3, num_hidden_layers=5):
        # 增加 hidden_dim 到 64 以增强表达能力
        super(DNN, self).__init__()
        layers = []
        # 使用 Siren 或 Tanh 均可，这里保持 Tanh 但建议后续尝试 Siren
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------------------------------------------------------------#

# R-PAG SOLVER IMPLEMENTATION (The Innovation)
class EulerRPAGSolver:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.8) # 学习率衰减
        
        # GCond: 历史梯度缓冲区
        self.V_smooth = {} 
        self.beta = 0.9 # 动量系数
        for name, param in model.named_parameters():
            self.V_smooth[name] = torch.zeros_like(param.data)

    def get_gradients(self, loss):
        """计算梯度但不更新"""
        grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True, allow_unused=True)
        grad_dict = {}
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            if grad is not None:
                grad_dict[name] = grad
            else:
                grad_dict[name] = torch.zeros_like(param)
        return grad_dict

    def asymmetric_projection(self, g_low, g_high):
        """R-PAG 核心：非对称投影"""
        g_low_flat = torch.cat([g.flatten() for g in g_low.values()])
        g_high_flat = torch.cat([g.flatten() for g in g_high.values()])
        
        dot = torch.dot(g_low_flat, g_high_flat)
        
        if dot < 0: # 冲突检测
            norm_sq = torch.dot(g_high_flat, g_high_flat) + 1e-8
            proj_coeff = dot / norm_sq
            
            g_low_new = {}
            for name in g_low:
                g_low_new[name] = g_low[name] - proj_coeff * g_high[name]
            return g_low_new, True
        return g_low, False

    def compute_loss_components(self, x_int, x_ic, rho_ic, u_ic, p_ic, eps_visc):
        """计算 Euler 残差（带人工粘性）和 IC Loss"""
        
        # 1. Forward Pass
        y = self.model(x_int)
        rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        gamma = 1.4

        # 2. Gradients (1st order)
        drho_dt_dx = compute_gradients(rho, x_int)
        rho_t, rho_x = drho_dt_dx[:, :1], drho_dt_dx[:, 1:]
        
        du_dt_dx = compute_gradients(u, x_int)
        u_t, u_x = du_dt_dx[:, :1], du_dt_dx[:, 1:]
        
        dp_dt_dx = compute_gradients(p, x_int)
        p_t, p_x = dp_dt_dx[:, :1], dp_dt_dx[:, 1:]

        # 3. Gradients (2nd order for Viscosity)
        rho_xx = compute_second_derivative(drho_dt_dx, x_int)
        u_xx = compute_second_derivative(du_dt_dx, x_int)
        p_xx = compute_second_derivative(dp_dt_dx, x_int) # Energy eq approx

        # 4. Euler Residuals with Artificial Viscosity
        # Mass: rho_t + (rho*u)_x = eps * rho_xx
        res_mass = rho_t + u * rho_x + rho * u_x - eps_visc * rho_xx
        
        # Momentum: rho*(u_t + u*u_x) + p_x = eps * u_xx
        res_mom = rho * (u_t + u * u_x) + p_x - eps_visc * u_xx
        
        # Energy: p_t + u*p_x + gamma*p*u_x = eps * p_xx
        res_eng = p_t + u * p_x + gamma * p * u_x - eps_visc * p_xx

        # 5. Dynamic Grouping (R-PAG Logic)
        # Calculate pointwise squared residuals
        pointwise_res = res_mass**2 + res_mom**2 + res_eng**2
        res_abs = torch.sqrt(pointwise_res).detach()
        
        # Threshold for Shock Group (Top 10%)
        threshold = torch.quantile(res_abs, 0.90)
        mask_shock = (res_abs >= threshold).float()
        mask_smooth = 1.0 - mask_shock
        
        # Weighted Losses
        L_shock = torch.sum(pointwise_res * mask_shock) / (torch.sum(mask_shock) + 1e-8)
        L_smooth = torch.sum(pointwise_res * mask_smooth) / (torch.sum(mask_smooth) + 1e-8)
        
        # 6. IC Loss
        y_ic = self.model(x_ic)
        loss_ic = ((y_ic[:, 0] - rho_ic)**2).mean() + \
                  ((y_ic[:, 1] - p_ic)**2).mean() + \
                  ((y_ic[:, 2] - u_ic)**2).mean()
                  
        return L_shock, L_smooth, loss_ic

    def step(self, x_int, x_ic, rho_ic, u_ic, p_ic, eps_visc):
        self.optimizer.zero_grad()
        
        # 1. 计算各部分 Loss
        L_shock, L_smooth, L_ic = self.compute_loss_components(x_int, x_ic, rho_ic, u_ic, p_ic, eps_visc)
        
        # 2. 梯度计算
        # High Priority: Shock + IC (IC 定义了解的物理约束，必须高优先)
        g_shock = self.get_gradients(L_shock * 20.0) # 激波权重加倍
        g_ic = self.get_gradients(L_ic * 100.0)      # IC 权重最高
        
        g_high = {}
        for name in g_shock:
            g_high[name] = g_shock[name] + g_ic[name]
            
        # Low Priority: Smooth
        g_smooth_raw = self.get_gradients(L_smooth * 1.0)
        
        # 3. GCond: 历史梯度平滑 (仅对平滑区)
        g_smooth_stable = {}
        for name in g_smooth_raw:
            self.V_smooth[name] = self.beta * self.V_smooth[name] + (1 - self.beta) * g_smooth_raw[name]
            g_smooth_stable[name] = self.V_smooth[name]
            
        # 4. Projection: 非对称投影
        g_smooth_final, is_conflict = self.asymmetric_projection(g_smooth_stable, g_high)
        
        # 5. Update Parameters
        for name, param in self.model.named_parameters():
            param.grad = g_high[name] + g_smooth_final[name]
            
        self.optimizer.step()
        self.scheduler.step()
        
        return (L_shock + L_smooth + L_ic).item(), is_conflict

# -----------------------------------------------------------------------------------------------------------------------------------#

# ANALYTICAL SOLUTION (Kept from base code)
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
        p_star = 0.3 # Fallback if optimization fails

    u_star = 0.5*(u_L + u_R) + 0.5*(f(p_star, p_R, rho_R, a_R) - f(p_star, p_L, rho_L, a_L))
    rho_star_L = rho_L * (p_star/p_L)**(1/gamma)
    rho_star_R = rho_R * ((p_star/p_R + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1)*p_star/p_R + 1))
    s_shock = u_R + a_R * np.sqrt((gamma+1)/(2*gamma)*(p_star/p_R) + (gamma-1)/(2*gamma))
    head_fan = u_L - a_L
    tail_fan = u_star - a_L * (p_star/p_L)**((gamma-1)/(2*gamma))

    rho_sol, p_sol, u_sol = np.zeros_like(x_np), np.zeros_like(x_np), np.zeros_like(x_np)
    xi = (x_np - x_0) / (t_val + 1e-10)

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
        axs[0, i].plot(x, exact_data[i], 'k--', label='Exact', linewidth=2)
        axs[0, i].plot(x, pinn_data[i], color=colors[i], label='R-PAG PINN', alpha=0.8)
        axs[0, i].set_title(f'{titles[i]} (t={t_eval})\nRel. L2 Err: {errors[i]:.4e}')
        axs[0, i].legend()
        axs[0, i].grid(True, alpha=0.3)

        abs_err = np.abs(pinn_data[i] - exact_data[i])
        axs[1, i].plot(x, abs_err, color=colors[i], linewidth=1)
        axs[1, i].fill_between(x, abs_err, color=colors[i], alpha=0.2)
        axs[1, i].set_title(f'{titles[i]} Absolute Error')
        axs[1, i].set_yscale('log')
        axs[1, i].grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------#

# MAIN FUNCTION (Modified for Curriculum Learning)
def main():
    # 1. Hyperparameters
    learning_rate = 0.001
    num_x = 1000
    num_t = 1000
    num_ic_samples = 1000
    num_f_samples = 10000

    # 2. Data Generation
    x = np.linspace(0, 1, num_x)
    t = np.linspace(0, 0.2, num_t)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]

    # IC Training Data
    ic_indices = np.random.choice(num_x, num_ic_samples, replace=False)
    x_ic = x_grid[ic_indices, 0][:, None]
    t_ic = t_grid[ic_indices, 0][:, None]
    x_ic_train_np = np.hstack((t_ic, x_ic))
    rho_ic_np, u_ic_np, p_ic_np = initial_conditions(x_ic.flatten())

    # Interior Training Data
    f_indices = np.random.choice(len(X), num_f_samples, replace=False)
    x_int_train_np = np.hstack((T[f_indices], X[f_indices]))

    # Test Data
    x_test_np = np.hstack((T, X))

    # To Tensor
    x_ic_train = torch.tensor(x_ic_train_np, dtype=torch.float32, device=device)
    rho_ic = torch.tensor(rho_ic_np, dtype=torch.float32, device=device)
    u_ic = torch.tensor(u_ic_np, dtype=torch.float32, device=device)
    p_ic = torch.tensor(p_ic_np, dtype=torch.float32, device=device)
    x_int_train = torch.tensor(x_int_train_np, dtype=torch.float32, requires_grad=True, device=device)
    x_test = torch.tensor(x_test_np, dtype=torch.float32, device=device)

    # 3. Model & R-PAG Solver
    model = DNN().to(device)
    solver = EulerRPAGSolver(model, lr=learning_rate)

    # 4. Curriculum Training Loop (Viscosity Continuation)
    # 阶段 1: 高粘性 (学习大轮廓) -> 阶段 2: 中粘性 -> 阶段 3: 低粘性 (逼近真实解)
    viscosity_schedule = [0.01, 0.005, 0.001] 
    epochs_schedule = [3000, 3000, 4000] # 总共 10000 epochs

    print(">>> Starting R-PAG Curriculum Training <<<")
    start_time = time.time()
    
    total_epochs_run = 0
    for stage, (eps, n_epochs) in enumerate(zip(viscosity_schedule, epochs_schedule)):
        print(f"\n--- Stage {stage+1}: Artificial Viscosity eps = {eps} ---")
        
        for epoch in range(1, n_epochs + 1):
            model.train()
            loss, is_conflict = solver.step(x_int_train, x_ic_train, rho_ic, u_ic, p_ic, eps)
            
            total_epochs_run += 1
            if total_epochs_run % 500 == 0:
                status = "CONFLICT (Projected)" if is_conflict else "OK"
                print(f"Epoch {total_epochs_run:05d} | Loss: {loss:.6f} | Status: {status}")

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

    # 5. Evaluation
    print("Evaluating model...")
    rho_pred, p_pred, u_pred = evaluate_model(model, x_test, num_x, num_t)
    
    # 6. Plotting
    plot_comparison_with_error(t, x, rho_pred, p_pred, u_pred, t_eval=0.2)

if __name__ == '__main__':
    main()