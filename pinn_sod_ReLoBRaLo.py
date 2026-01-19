"""
Supervisor: Prof. K. Hejranfar
Author: Mohammad E. Heravifard
Project #01: Data-Free Physics-Driven Neural Networks Approach
for 1D Euler Equations for Compressible Flows:
Testcase #01: Sod's Shock Tube Problem.
Method: ReLoBRaLo Implementation
"""
# -----------------------------------------------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from scipy.optimize import brentq
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------------------------------------------#

# SET RANDOM SEEDS
torch.manual_seed(123456)
np.random.seed(123456)

# -------------------------------------------------------------------------------------------------------------------------------------#

class ReLoBRaLoBalancer:
    """
    ReLoBRaLo (Relative Loss Balancing with Random Lookback) implementation for PyTorch.
    Based on Bischof et al. (2021).
    """
    def __init__(self, num_losses, T=0.1, alpha=0.999, rho=0.99):
        """
        Args:
            num_losses (int): 需要平衡的损失项数量.
            T (float): Softmax 温度参数.
            alpha (float): 权重的指数移动平均衰减率 (0.9-0.999).
            rho (float): 随机回溯的伯努利概率 (Bernoulli probability).
        """
        self.num_losses = num_losses
        self.T = T
        self.alpha = alpha
        self.rho_prob = rho 
        
        # 初始化权重为全 1 (默认在 CPU，稍后会自动迁移)
        self.lambdas = torch.ones(num_losses, requires_grad=False)
        
        # 历史记录
        self.last_losses = None 
        self.init_losses = None

    def update_weights(self, current_losses_tensor):
        """
        根据当前损失计算新的权重 lambda。
        Args:
            current_losses_tensor (torch.Tensor): 形状为 [num_losses] 的当前损失张量
        Returns:
            torch.Tensor: 更新后的权重 (detach graph)
        """
        # [关键修复]：检测并确保 lambdas 在正确的设备上 (如 cuda:0)
        if self.lambdas.device != current_losses_tensor.device:
            self.lambdas = self.lambdas.to(current_losses_tensor.device)

        # 第一次迭代初始化
        if self.last_losses is None:
            self.last_losses = current_losses_tensor.detach()
            self.init_losses = current_losses_tensor.detach()
            return self.lambdas

        # 1. 计算短期相对变化
        denominator = self.last_losses * self.T + 1e-12
        ratio_short = current_losses_tensor.detach() / denominator
        lambdas_hat_short = F.softmax(ratio_short, dim=0) * self.num_losses

        # 2. 计算长期(回溯)相对变化
        denominator_0 = self.init_losses * self.T + 1e-12
        ratio_long = current_losses_tensor.detach() / denominator_0
        lambdas_hat_long = F.softmax(ratio_long, dim=0) * self.num_losses

        # 3. 确定是否进行回溯
        rho_sample = 1.0 if np.random.rand() < self.rho_prob else 0.0

        # 4. 更新权重 lambda
        # 所有参与运算的 tensor 现在都在同一个设备上了
        new_lambdas = (self.rho_prob * self.alpha * self.lambdas + 
                       (1 - self.rho_prob) * self.alpha * lambdas_hat_long + 
                       (1 - self.alpha) * lambdas_hat_short)
        
        # 更新内部状态
        self.lambdas = new_lambdas.detach()
        self.last_losses = current_losses_tensor.detach()
        
        if rho_sample == 0.0:
            self.init_losses = current_losses_tensor.detach()

        return self.lambdas

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

    def get_pde_residuals(self, x):
        """
        计算 PDE 的各个残差分量，不再求和返回。
        Returns:
            list of torch.Tensor: [mass_res, mom_res, energy_res]
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

        # Euler equations residuals
        mass_residual = rho_t + u * rho_x + rho * u_x
        momentum_residual = rho * (u_t + u * u_x) + p_x
        energy_residual = p_t + gamma * p * u_x + u * p_x

        # 返回每个方程的 MSE
        loss_mass = (mass_residual**2).mean()
        loss_mom = (momentum_residual**2).mean()
        loss_energy = (energy_residual**2).mean()

        return [loss_mass, loss_mom, loss_energy]

    def get_ic_errors(self, x_ic, rho_ic, u_ic, p_ic):
        """
        计算 IC 的各个误差分量，不再求和返回。
        Returns:
            list of torch.Tensor: [rho_err, p_err, u_err]
        """
        y_ic = self.net(x_ic)
        rho_ic_nn, p_ic_nn, u_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2]

        loss_rho = ((rho_ic_nn - rho_ic) ** 2).mean()
        loss_p = ((p_ic_nn - p_ic) ** 2).mean()
        loss_u = ((u_ic_nn - u_ic) ** 2).mean()

        return [loss_rho, loss_p, loss_u]

# -----------------------------------------------------------------------------------------------------------------------------------#

# UTILITY FUNCTIONS
def compute_gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True)[0]

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

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

def train_model(model, optimizer, scheduler, x_int_train, x_ic_train, rho_ic, u_ic, p_ic, epochs, device):
    """
    使用 ReLoBRaLo 算法训练模型
    """
    # 初始化 ReLoBRaLo 平衡器
    # 我们有 6 个损失项: [Mass, Mom, Energy, IC_Rho, IC_P, IC_U]
    # T=0.1, alpha=0.999 是原论文推荐的默认值
    balancer = ReLoBRaLoBalancer(num_losses=6, T=0.1, alpha=0.999, rho=0.99)
    
    # 用于记录 loss 曲线
    loss_history = []
    weights_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 1. 计算所有独立损失项
        pde_losses = model.get_pde_residuals(x_int_train) # [mass, mom, energy]
        ic_losses = model.get_ic_errors(x_ic_train, rho_ic, u_ic, p_ic) # [rho, p, u]
        
        # 合并所有损失到一个列表
        all_losses_list = pde_losses + ic_losses # 长度为 6 的列表
        
        # 堆叠成 tensor 用于计算权重 (不需要梯度用于权重计算)
        current_losses_tensor = torch.stack(all_losses_list)
        
        # 2. ReLoBRaLo 更新权重
        # 注意：这里我们使用 detach() 的 loss 来计算权重，避免计算图循环引用
        weights = balancer.update_weights(current_losses_tensor)
        
        # 3. 计算加权总损失
        # total_loss = sum(w_i * L_i)
        total_loss = torch.sum(weights * current_losses_tensor)

        # 记录
        loss_val = total_loss.item()
        loss_history.append(loss_val)
        weights_history.append(weights.cpu().numpy())

        if epoch % 100 == 0:
            w_str = ", ".join([f"{w:.2f}" for w in weights])
            l_str = ", ".join([f"{l:.4e}" for l in current_losses_tensor])
            print(f"Epoch {epoch:04d}: Total Loss = {loss_val:.6f}")
            print(f"    Weights: [{w_str}]")
            print(f"    Losses : [{l_str}]")

        # 4. 反向传播
        total_loss.backward()
        optimizer.step()
        scheduler.step()
    
    # 训练结束后绘制权重变化（可选）
    plot_weights_history(weights_history)

def plot_weights_history(weights_history):
    weights_np = np.array(weights_history)
    labels = ['PDE Mass', 'PDE Mom', 'PDE Energy', 'IC Rho', 'IC P', 'IC U']
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(weights_np[:, i], label=labels[i])
    plt.xlabel('Epochs')
    plt.ylabel('Loss Weights $\lambda$')
    plt.title('ReLoBRaLo Adaptive Weights Evolution')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------#
# (Evaluate functions remain mostly the same, included below for completeness)

def get_exact_sod(x_np, t_val, gamma=1.4):
    # Same as original implementation
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
    rho_pinn, p_pinn, u_pinn = rho_pred[:, t_idx], p_pred[:, t_idx], u_pred[:, t_idx]
    rho_ex, p_ex, u_ex = get_exact_sod(x, t_eval)
    def calc_l2(pred, exact): return np.sqrt(np.mean((pred - exact)**2)) / np.sqrt(np.mean(exact**2))
    err_rho, err_p, err_u = calc_l2(rho_pinn, rho_ex), calc_l2(p_pinn, p_ex), calc_l2(u_pinn, u_ex)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    titles = ['Density', 'Pressure', 'Velocity']
    pinn_data = [rho_pinn, p_pinn, u_pinn]
    exact_data = [rho_ex, p_ex, u_ex]
    errors = [err_rho, err_p, err_u]
    colors = ['blue', 'red', 'green']
    for i in range(3):
        axs[0, i].plot(x, exact_data[i], 'k--', label='Exact Solution', linewidth=2)
        axs[0, i].plot(x, pinn_data[i], color=colors[i], label='PINN (ReLoBRaLo)', alpha=0.7)
        axs[0, i].set_title(f'{titles[i]} at t={t_eval}\nRel. L2 Error: {errors[i]:.4e}')
        axs[0, i].legend()
        axs[0, i].grid(True, alpha=0.3)
        abs_err = np.abs(pinn_data[i] - exact_data[i])
        axs[1, i].fill_between(x, abs_err, color=colors[i], alpha=0.2)
        axs[1, i].plot(x, abs_err, color=colors[i], linewidth=1)
        axs[1, i].set_title(f'{titles[i]} Absolute Error')
        axs[1, i].set_yscale('log')
        axs[1, i].grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------#

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    learning_rate = 0.001
    epochs = 5000 # 可以适当减少 epoch，因为平衡算法通常收敛更快
    num_x = 2000
    num_t = 1000
    num_ic_samples = 2000
    num_f_samples = 20000 

    # Grid Generation
    x = np.linspace(0, 1, num_x)
    t = np.linspace(0, 0.2, num_t)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]

    # Training Data
    ic_indices = np.random.choice(num_x, num_ic_samples, replace=False)
    x_ic = x_grid[ic_indices, 0][:, None]
    t_ic = t_grid[ic_indices, 0][:, None]
    x_ic_train = np.hstack((t_ic, x_ic))
    rho_ic_train_np, u_ic_train_np, p_ic_train_np = initial_conditions(x_ic.flatten())

    f_indices = np.random.choice(num_x * num_t, num_f_samples, replace=False)
    x_int_train = np.hstack((T[f_indices], X[f_indices]))
    x_test = np.hstack((T, X))

    # To Tensor
    x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32, device=device)
    x_int_train = torch.tensor(x_int_train, dtype=torch.float32, requires_grad=True, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    rho_ic_train = torch.tensor(rho_ic_train_np, dtype=torch.float32, device=device)
    u_ic_train = torch.tensor(u_ic_train_np, dtype=torch.float32, device=device)
    p_ic_train = torch.tensor(p_ic_train_np, dtype=torch.float32, device=device)

    # Model Setup
    model = DNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.8)

    print("Starting training with ReLoBRaLo...")
    start_time = time.time()
    train_model(model, optimizer, scheduler, x_int_train, x_ic_train,
                rho_ic_train, u_ic_train, p_ic_train, epochs, device)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    print("Evaluating model...")
    rho_pred, p_pred, u_pred = evaluate_model(model, x_test, num_x, num_t)
    plot_comparison_with_error(t, x, rho_pred, p_pred, u_pred, t_eval=0.2)

if __name__ == '__main__':
    main()