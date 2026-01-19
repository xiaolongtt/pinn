import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# 设置随机种子以保证结果可复现
torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================
# 1. 核心工具：数值求解与误差绘图
# ==========================================

def solve_burgers_numerical(t_target, nu=0.01/np.pi, num_x=1000):
    """
    使用有限差分法计算 Burgers 方程的参考解 (Ground Truth)
    """
    x = np.linspace(-1, 1, num_x)
    dx = x[1] - x[0]
    u0 = -np.sin(np.pi * x)

    def deriv(u, t):
        dudt = np.zeros_like(u)
        # 边界 u(-1)=0, u(1)=0，dudt边界保持0
        
        # 内部点：粘性项 (中心差分) + 对流项 (中心差分)
        u_xx = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        u_x  = (u[2:] - u[:-2]) / (2*dx)
        
        dudt[1:-1] = nu * u_xx - u[1:-1] * u_x
        return dudt

    t_eval = np.linspace(0, t_target, 100)
    sol = odeint(deriv, u0, t_eval)
    return x, sol[-1]

def plot_comparison_and_error(model, t_val=0.5):
    """
    绘制对比图，并显式标注误差
    """
    # 1. 获取数值参考解
    print(f"Calculating numerical reference for t={t_val}...")
    x_true, u_true = solve_burgers_numerical(t_target=t_val)
    
    # 2. 获取模型预测解
    x_tensor = torch.tensor(x_true.reshape(-1, 1), dtype=torch.float32)
    t_tensor = torch.full((len(x_true), 1), t_val, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(t_tensor, x_tensor).numpy().flatten()

    # 3. 计算误差指标
    # 相对 L2 误差
    l2_error = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    # 绝对误差
    abs_error = np.abs(u_true - u_pred)

    print(f"Time: {t_val}s | Relative L2 Error: {l2_error:.4e}")

    # 4. 绘图 (双子图)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左图：解的对比 ---
    axs[0].plot(x_true, u_true, 'k-', lw=2, label='Numerical Exact')
    axs[0].plot(x_true, u_pred, 'r--', lw=2, label='PINN Prediction')
    axs[0].set_xlabel('x', fontsize=12)
    axs[0].set_ylabel(f'u(t={t_val}, x)', fontsize=12)
    # >>> 关键修改：在标题中标注误差 <<<
    axs[0].set_title(f'Solution Comparison (t={t_val})\nRel L2 Error: {l2_error:.4e}', fontsize=14, color='blue')
    axs[0].legend(fontsize=10)
    axs[0].grid(True, alpha=0.5)

    # --- 右图：绝对误差分布 ---
    axs[1].plot(x_true, abs_error, 'b-', lw=1.5)
    axs[1].fill_between(x_true, abs_error, color='blue', alpha=0.1)
    axs[1].set_xlabel('x', fontsize=12)
    axs[1].set_ylabel('Absolute Error |u_true - u_pred|', fontsize=12)
    axs[1].set_title('Absolute Error Distribution', fontsize=14)
    axs[1].set_yscale('log') # 使用对数坐标查看微小误差
    axs[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_loss(losses):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, color='blue', lw=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==========================================
# 2. 模型与训练逻辑
# ==========================================

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
    
    def forward(self, t, x):
        return self.layer(torch.cat([t, x], dim=1))

def physics_loss(model, t, x):
    u = model(t, x)
    u_t = torch.autograd.grad(u, t, torch.ones_like(t), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(x), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(x), create_graph=True)[0]
    nu = 0.01 / torch.pi
    return (u_t + u * u_x - nu * u_xx).pow(2).mean()

def train(model, optimizer, num_epochs):
    print(f"Starting training for {num_epochs} epochs...")
    losses = []
    
    # 定义固定的边界和初始点采样 (也可以放在循环里随机采样)
    t_bc = torch.rand(500, 1)
    x_left = -torch.ones(500, 1)
    x_right = torch.ones(500, 1)
    x_ic = torch.rand(1000, 1) * 2 - 1
    t_0 = torch.zeros_like(x_ic)
    u_exact_ic = -torch.sin(torch.pi * x_ic)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 内部点随机采样
        t = torch.rand(2000, 1, requires_grad=True)
        x = torch.rand(2000, 1, requires_grad=True) * 2 - 1
        
        # 计算 Loss
        f_loss = physics_loss(model, t, x)
        
        # 边界 Loss
        u_left = model(t_bc, x_left)
        u_right = model(t_bc, x_right)
        bc_loss = u_left.pow(2).mean() + u_right.pow(2).mean()
        
        # 初始 Loss
        u_init = model(t_0, x_ic)
        ic_loss = (u_init - u_exact_ic).pow(2).mean()
        
        loss = f_loss + bc_loss + ic_loss
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
            
    return losses

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    # 初始化
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练 (建议 15000 轮左右，Burgers方程收敛需要一定时间)
    num_epochs = 15000 
    losses = train(model, optimizer, num_epochs)

    # 1. 绘制 Loss
    plot_loss(losses)

    # 2. 绘制 t=0.5 的详细误差分析图 (带误差标注)
    plot_comparison_and_error(model, t_val=0.5)